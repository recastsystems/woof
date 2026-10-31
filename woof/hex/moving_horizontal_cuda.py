"""Moving surface/soil donor sums in the CSR reference's exact row order."""
from functools import lru_cache
import threading

import numpy as np
from .moving_cuda_work import independent_stream

OPTIONS = ('--fmad=false',)
_GEOMETRY_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    extern "C" __global__ void horizontal(
        const long long *ptr,const long long *ids,const double *area,
        const double *target_area,const double *values,double *out,
        const long long ns,const long long nt,const long long products) {
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=nt*products)return;
        long long row=i%nt,product=i/nt;
        double value=0.;
        for(long long j=ptr[row];j<ptr[row+1];++j)
            value=__dadd_rn(value,__dmul_rn(area[j],values[product*ns+ids[j]]));
        out[i]=__ddiv_rn(value,target_area[row]);
    }
    ''','horizontal',options=OPTIONS)


@independent_stream
def intensive(weights, values):
    import cupy as cp
    a=np.asarray(values)
    if a.shape[-1]!=len(weights.source_area) or not np.all(np.isfinite(a)):
        raise ValueError('source cell extent or finite-value gate failed')
    matrix=weights.overlap
    # A caller may use the same geometry from concurrent independent fields.
    # Publish only after the upload stream finishes, before another thread sees it.
    with _GEOMETRY_LOCK:
        cached=getattr(weights,'_moving_gpu_horizontal',None)
        key=(id(matrix),int(cp.cuda.Device().id))
        if cached is None or cached[0]!=key:
            csr=matrix if getattr(matrix,'format',None)=='csr' else matrix.tocsr()
            geometry=tuple(cp.asarray(np.ascontiguousarray(x,dtype=dtype)) for x,dtype in
                ((csr.indptr,np.int64),(csr.indices,np.int64),(csr.data,np.float64),
                 (weights.target_area,np.float64)))
            cp.cuda.get_current_stream().synchronize()
            weights._moving_gpu_horizontal=(key,geometry)
        else:
            geometry=cached[1]
    flat=np.ascontiguousarray(a,dtype=np.float64).reshape(-1,a.shape[-1])
    source=cp.asarray(flat)
    target=cp.empty((flat.shape[0],len(weights.target_area)),dtype=cp.float64)
    if target.size:
        _kernel()(((target.size+255)//256,),(256,),
                  (*geometry,source,target,np.int64(flat.shape[1]),
                   np.int64(target.shape[1]),np.int64(target.shape[0])))
    return cp.asnumpy(target).reshape((*a.shape[:-1],len(weights.target_area)))
