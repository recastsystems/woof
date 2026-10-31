"""Installed NumPy-compatible double powers for moving thermodynamics.

Use the shared port of the pinned glibc algorithm, never CUDA's approximate
pow. Explicit rounded operators preserve the host expressions' grouping.
"""
from functools import lru_cache
import numpy as np

OPTIONS=('--fmad=false','--prec-div=true','--prec-sqrt=true')


@lru_cache(maxsize=1)
def kernels():
    import cupy as cp
    from .mpasphys.diagnostics_v842 import native_diagnostic_source
    code=native_diagnostic_source()+r'''
    extern "C" __global__ void moving_exner_pressure(
        const double *zz,const double *heat,double *exner,double *pressure,
        const double factor,const double exponent,const double inverse,const long long n){
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
        double e=glibc_pow(__dmul_rn(__dmul_rn(zz[i],factor),heat[i]),exponent);
        exner[i]=e;pressure[i]=__dmul_rn(1.e5,glibc_pow(e,inverse));
    }
    extern "C" __global__ void moving_balance_pressure(
        const double *zz,const double *heat,double *pressure,const long long n){
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
        double base=__ddiv_rn(__dmul_rn(__dmul_rn(287.,zz[i]),heat[i]),1.e5);
        pressure[i]=__dmul_rn(1.e5,glibc_pow(base,1.4));
    }
    extern "C" __global__ void moving_ground_extension(
        const long long *cols,const double *height,const double *ground,const double *centre,
        const double *theta,const double *exner,const double *loading,double *out,
        const long long nnz,const int levels){
        long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(j>=nnz)return;
        long long c=cols[j];
        double rate=__ddiv_rn(__dmul_rn(9.80616,loading[c]),__dmul_rn(1004.5,theta[c]));
        double base=__dadd_rn(exner[c],__dmul_rn(rate,__dsub_rn(centre[c],ground[c])));
        double gp=__dmul_rn(1.e5,glibc_pow(base,3.5));
        for(int k=0;k<levels;++k){
            long long i=(long long)k*nnz+j;
            double z=height[i]<ground[c]?height[i]:ground[c];
            double b=__dadd_rn(exner[c],__dmul_rn(rate,__dsub_rn(centre[c],z)));
            double p=__dmul_rn(1.e5,glibc_pow(b,3.5));
            out[i]=__ddiv_rn(__dsub_rn(p,gp),__dmul_rn(9.80616,loading[c]));
        }
    }
    '''
    module=cp.RawModule(code=code,options=OPTIONS)
    return (module.get_function('moving_exner_pressure'),module.get_function('moving_balance_pressure'),
            module.get_function('moving_ground_extension'))


def exner_pressure(zz,heat):
    import cupy as cp
    from .moving_balance import DRY_RGAS,DRY_CP
    exner,pressure=cp.empty_like(heat),cp.empty_like(heat)
    kernels()[0](((heat.size+127)//128,),(128,),(
        zz,heat,exner,pressure,np.float64(DRY_RGAS/1.e5),np.float64(DRY_RGAS/(DRY_CP-DRY_RGAS)),
        np.float64(DRY_CP/DRY_RGAS),np.int64(heat.size)))
    return exner,pressure


def balance_pressure(zz,heat,pressure):
    kernels()[1](((heat.size+127)//128,),(128,),(zz,heat,pressure,np.int64(heat.size)))


from .moving_cuda_work import independent_stream


@independent_stream
def ground_extension(weights,theta,exner,loading):
    import cupy as cp
    from .moving_vertical_cuda import _cumulative_geometry
    arrays=_cumulative_geometry(weights)['arrays']
    inputs=tuple(cp.asarray(np.ascontiguousarray(x)) for x in (theta,exner,loading))
    out=cp.empty(weights.sample_height.shape,dtype=cp.float64)
    nnz=out.shape[1]
    kernels()[2](((nnz+127)//128,),(128,),(
        arrays[1],arrays[4],arrays[5],arrays[6],*inputs,out,np.int64(nnz),np.int32(out.shape[0])))
    host=cp.asnumpy(out)
    return host,dict(device=cp.cuda.runtime.getDevice(),source=host,values=out)
