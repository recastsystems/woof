"""Moving saturation Newton iterations with NumPy-compatible device powers."""
from functools import lru_cache
import os
import numpy as np
from .moving_cuda_work import independent_stream

OPTIONS = ('--fmad=false',)


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    code = r'''
    __device__ double A(double a,double b){return __dadd_rn(a,b);}
    __device__ double S(double a,double b){return __dsub_rn(a,b);}
    __device__ double M(double a,double b){return __dmul_rn(a,b);}
    __device__ double D(double a,double b){return __ddiv_rn(a,b);}
    __device__ double lo(double a,double b){return a<b?a:b;}
    __device__ double hi(double a,double b){return a>b?a:b;}
    __device__ double saturation(double t,double p){
        const double x=hi(-80.,S(t,273.16));
        double e=-.321582393e-13;
        e=A(.379534310e-11,M(x,e));
        e=A(.702620698e-8,M(x,e));
        e=A(.203154182e-5,M(x,e));
        e=A(.299291081e-3,M(x,e));
        e=A(.264224321e-1,M(x,e));
        e=A(.143177157e01,M(x,e));
        e=A(.444606896e02,M(x,e));
        e=A(.611583699e03,M(x,e));
        e=lo(e,M(p,.15));
        return D(M(.622,e),hi(1.e-4,S(p,e)));
    }
    extern "C" __global__ void evaporation(
        const double *rho,const double *heat,const double *vapor,const double *cloud,
        const double *exner,const double *pressure,double *out,
        const double vapor_ratio,const double latent_ratio,const long long n){
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=n)return;
        const double c=hi(cloud[i],0.);
        const double t=M(D(D(heat[i],rho[i]),A(1.,M(vapor_ratio,vapor[i]))),exner[i]);
        double x=0.;
        for(int it=0;it<4;++it){
            const double cooled=S(t,M(latent_ratio,x));
            const double sat=saturation(cooled,pressure[i]);
            const double slope=D(S(saturation(A(cooled,.01),pressure[i]),sat),.01);
            x=S(x,D(S(A(vapor[i],x),sat),A(1.,M(latent_ratio,slope))));
            x=lo(hi(x,0.),c);
        }
        out[i]=x;out[n+i]=t;
    }
    '''
    return cp.RawKernel(code, 'evaporation', options=OPTIONS)


def _device_evaporation(rho, heat, zz, vapor, cloud):
    import cupy as cp
    from .moving_balance import DRY_RGAS, DRY_CP, VAPOR_RGAS, LATENT_HEAT
    rho, heat, zz, vapor, cloud = (np.asarray(x, dtype=np.float64) for x in (rho, heat, zz, vapor, cloud))
    arrays = tuple(cp.asarray(np.ascontiguousarray(x)) for x in (rho, heat, vapor, cloud))
    if os.environ.get('WOOF_HEX_MOVING_POW_GPU','1') != '0':
        from .moving_math_cuda import exner_pressure
        dz=cp.asarray(np.ascontiguousarray(zz))
        exner,pressure=exner_pressure(dz,arrays[1])
    else:
        exner=(zz*(DRY_RGAS/1.e5)*heat)**(DRY_RGAS/(DRY_CP-DRY_RGAS))
        pressure=1.e5*exner**(DRY_CP/DRY_RGAS)
        exner,pressure=(cp.asarray(np.ascontiguousarray(x)) for x in (exner,pressure))
    arrays += exner,pressure
    result = cp.empty((2, *rho.shape), dtype=cp.float64)
    _kernel()(((rho.size+127)//128,), (128,), (*arrays, result,
        np.float64(VAPOR_RGAS/DRY_RGAS), np.float64(LATENT_HEAT/DRY_CP), np.int64(rho.size)))
    return arrays, result, exner


@independent_stream
def evaporation(rho, heat, zz, vapor, cloud):
    import cupy as cp
    _, result, exner = _device_evaporation(rho,heat,zz,vapor,cloud)
    host = cp.asnumpy(result)
    return host[0],host[1],cp.asnumpy(exner)


@lru_cache(maxsize=1)
def _adjust_kernel():
    import cupy as cp
    source = r'''
    extern "C" __global__ void adjust(
        const double *rho,const double *heat,const double *vapor,const double *cloud,
        const double *donor,const double *exner,const double *evaporated,
        double *out,const double threshold,const double latent,const double ratio,const long long n){
        long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
        double moved=__dsub_rn(__dmul_rn(evaporated[i],rho[i]),donor[i]);
        const double cl=cloud[i]>0.?cloud[i]:0.;
        moved=moved>__dmul_rn(threshold,rho[i])?(moved<cl?moved:cl):0.;
        double q=__dadd_rn(__ddiv_rn(vapor[i],rho[i]),__ddiv_rn(moved,rho[i]));
        double t=evaporated[n+i];
        double cooled=__dsub_rn(t,__ddiv_rn(__dmul_rn(latent,moved),rho[i]));
        double h=__dmul_rn(__ddiv_rn(__dmul_rn(rho[i],cooled),exner[i]),
                           __dadd_rn(1.,__dmul_rn(ratio,q)));
        out[i]=__dadd_rn(vapor[i],moved);
        out[n+i]=__dsub_rn(cloud[i],moved);
        out[2*n+i]=moved>0.?h:heat[i];
        out[3*n+i]=moved;
        out[4*n+i]=__dsub_rn(t,cooled);
    }
    '''
    return cp.RawKernel(source,'adjust',options=OPTIONS)


@independent_stream
def adjustment(vapor,cloud,rho,heat,zz,donor,total,threshold):
    """Keep phase transfer and latent cooling beside evaporation on the card.

    NumPy still owns both power calls and the ordered integral reductions.
    Every pointwise operator uses the original FP64 evaluation order.
    """
    import cupy as cp
    from .moving_balance import DRY_RGAS,DRY_CP,VAPOR_RGAS,LATENT_HEAT
    fixed,evaporated,_ = _device_evaporation(rho,heat,zz,vapor/rho,cloud/rho)
    dv,dc,dd = (cp.asarray(np.ascontiguousarray(x,dtype=np.float64)) for x in (vapor,cloud,donor))
    result = cp.empty((5,*rho.shape),dtype=cp.float64)
    _adjust_kernel()(((rho.size+127)//128,),(128,),(
        fixed[0],fixed[1],dv,dc,dd,fixed[4],evaporated,result,np.float64(threshold),
        np.float64(LATENT_HEAT/DRY_CP),np.float64(VAPOR_RGAS/DRY_RGAS),np.int64(rho.size)))
    moved = result[3]
    positive = moved>0
    stats = cp.asnumpy(cp.stack((cp.count_nonzero(cp.any(positive,axis=0)),cp.count_nonzero(positive),
                                  cp.maximum(cp.max(moved/fixed[0]),0.),cp.max(result[4]))))
    host = cp.asnumpy(result[:4]) if stats[1] else None
    transferred = float(total(cp.asnumpy(moved) if host is None else host[3]))
    receipt = dict(method="target saturation adjustment of the cloud water the remap left subsaturated beyond "
                          "the donors' own; latent cooling applied before the balanced initialization",
                   cells=int(stats[0]),layers=int(stats[1]),cloud_to_vapor_kg=transferred,
                   cloud_total_kg=float(total(cloud)),
                   largest_mixing_ratio=float(stats[2]))
    if not stats[1]:
        receipt.update(theta_mass_change_kg_kelvin=0.,largest_cooling_k=0.)
        return vapor,cloud,heat,receipt
    receipt.update(theta_mass_change_kg_kelvin=float(total(host[2]-heat)),largest_cooling_k=float(stats[3]))
    return host[0],host[1],host[2],receipt
