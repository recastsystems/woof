"""Ordered overlapping surface-role donor selection; nearest fallback stays exact."""
from functools import lru_cache
import numpy as np
from .moving_cuda_work import independent_stream, shared_geometry


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.RawKernel(r'''
    extern "C" __global__ void surface_donors(
        const long long *ptr,const long long *cols,const double *area,
        const long long *source,const long long *target,long long *donors,
        signed char *labels,const int count) {
        int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;
        long long old=donors[row];bool same=true;
        for(int f=0;f<5;++f)if(source[old*5+f]!=target[(long long)row*5+f])same=false;
        labels[row]=-2;if(same)return;
        labels[row]=-1;
        for(int tier=0;tier<4;++tier) {
            long long best=-1;double weight=0.;
            for(long long j=ptr[row];j<ptr[row+1];++j) {
                long long candidate=cols[j];bool match=true;
                for(int f=0;f<5;++f) {
                    if((tier==1 && f==3)||(tier==2 && f==2)||(tier==3 && (f==2||f==3)))continue;
                    if(source[candidate*5+f]!=target[(long long)row*5+f])match=false;
                }
                // Strict greater preserves the first CSR donor on equal areas.
                if(match && (best<0 || area[j]>weight)){best=candidate;weight=area[j];}
            }
            if(best>=0){donors[row]=best;labels[row]=tier;return;}
        }
    }
    ''','surface_donors')


@shared_geometry('_cuda_surface_geometry')
def _geometry(weights):
    import cupy as cp
    cache=getattr(weights,'_cuda_surface_geometry',None)
    device=int(cp.cuda.Device().id)
    if cache is None or cache['device']!=device:
        csr=weights.overlap.tocsr()
        cache=dict(device=device,arrays=tuple(cp.asarray(np.ascontiguousarray(x,dtype=dtype))
            for x,dtype in ((csr.indptr,np.int64),(csr.indices,np.int64),(csr.data,np.float64))))
        weights._cuda_surface_geometry=cache
    return cache


@independent_stream
def overlap_donors(weights,source_roles,target_roles,initial):
    import cupy as cp
    count=len(initial)
    if source_roles.shape!=(len(weights.source_area),5) or target_roles.shape!=(count,5):
        raise ValueError('surface role extents differ from moving overlap')
    source,target,donors=(cp.asarray(np.ascontiguousarray(a,dtype=np.int64))
                           for a in (source_roles,target_roles,initial))
    labels=cp.empty(count,dtype=cp.int8)
    geometry=_geometry(weights)
    if count:_kernel()(((count+127)//128,),(128,),(*geometry['arrays'],source,target,donors,labels,np.int32(count)))
    return cp.asnumpy(donors),cp.asnumpy(labels)
