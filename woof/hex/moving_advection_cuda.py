"""Native advection stencil construction for the admitted moving class mesh."""
from functools import lru_cache
import numpy as np
from .moving_cuda_work import independent_stream

OPTIONS=('--fmad=false',)


@lru_cache(maxsize=4)
def _kernel(dtype,width):
    import cupy as cp
    scalar='float' if np.dtype(dtype)==np.dtype('float32') else 'double'
    code=r'''
    #define T SCALAR
    #define W WIDTH
    __device__ float A(float x,float y){return __fadd_rn(x,y);}
    __device__ double A(double x,double y){return __dadd_rn(x,y);}
    __device__ float S(float x,float y){return __fsub_rn(x,y);}
    __device__ double S(double x,double y){return __dsub_rn(x,y);}
    __device__ float P(float x,float y){return __fmul_rn(x,y);}
    __device__ double P(double x,double y){return __dmul_rn(x,y);}
    __device__ float D(float x,float y){return __fdiv_rn(x,y);}
    __device__ double D(double x,double y){return __ddiv_rn(x,y);}
    extern "C" __global__ void advection(
        const long long *counts,const long long *neighbors,const long long *owners,
        const T *deriv,const T *dc,const T *dv,T *adv,T *adv3,long long *cells,
        long long *number,int *failure,const int ne,const int slots) {
        int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=ne)return;
        long long base=(long long)e*W;
        for(int p=0;p<W;++p){cells[base+p]=-1;adv[base+p]=T(0);adv3[base+p]=T(0);}
        cells[base]=owners[(long long)e*2];cells[base+1]=owners[(long long)e*2+1];
        int filled=2;
        for(int side=0;side<2;++side) {
            long long owner=owners[(long long)e*2+side];
            for(int slot=0;slot<counts[owner];++slot) {
                long long cell=neighbors[owner*slots+slot];bool present=false;
                for(int p=0;p<filled;++p)if(cells[base+p]==cell)present=true;
                if(!present) {
                    if(filled>=W){failure[e]=1;return;}
                    cells[base+filled]=cell;++filled;
                }
            }
        }
        number[e]=filled;
        for(int side=0;side<2;++side) {
            long long owner=owners[(long long)e*2+side];
            for(int slot=0;slot<=counts[owner];++slot) {
                long long cell=slot==0?owner:neighbors[owner*slots+slot-1];
                int position=0;for(int p=0;p<filled;++p)if(cells[base+p]==cell){position=p;break;}
                T value=deriv[((long long)e*2+side)*W+slot];
                adv[base+position]=A(adv[base+position],value);
                adv3[base+position]=side==0?A(adv3[base+position],value):S(adv3[base+position],value);
            }
        }
        T squared=P(dc[e],dc[e]);
        for(int p=0;p<filled;++p) {
            adv[base+p]=D(P(-squared,adv[base+p]),T(12));
            adv3[base+p]=D(P(-squared,adv3[base+p]),T(12));
        }
        // Each owner addition is a separate rounding point, even if a caller
        // eventually admits an edge whose owners coincide.
        for(int side=0;side<2;++side) {
            long long owner=owners[(long long)e*2+side];
            int position=0;for(int p=0;p<filled;++p)if(cells[base+p]==owner){position=p;break;}
            adv[base+position]=A(adv[base+position],T(.5));
        }
        for(int p=0;p<W;++p){adv[base+p]=P(adv[base+p],dv[e]);adv3[base+p]=P(adv3[base+p],dv[e]);}
    }
    '''.replace('SCALAR',scalar).replace('WIDTH',str(int(width)))
    return cp.RawKernel(code,'advection',options=OPTIONS)


@independent_stream
def build(mesh,*,config_scalar_adv_order,n_vert_levels):
    import cupy as cp
    from .transport import (_validate_order,_topology,_mesh_array,_mesh_value,_normalize_deriv_two,
                            _float_array,build_advection_coefficients,AdvectionCoefficients)
    order=_validate_order(config_scalar_adv_order,'config_scalar_adv_order')
    nc,ne,counts,edges,neighbors,owners=_topology(mesh)
    deriv=_normalize_deriv_two(_mesh_array(mesh,'deriv_two'),ne)
    dtype=deriv.dtype
    dc=_float_array(_mesh_array(mesh,'dcEdge'),'dcEdge',dtype)
    dv=_float_array(_mesh_array(mesh,'dvEdge'),'dvEdge',dtype)
    if dc.shape!=(ne,) or dv.shape!=(ne,):raise ValueError('dcEdge and dvEdge must have one value per edge')
    ids_value=_mesh_value(mesh,'indexToCellID',None)
    ids=np.arange(nc,dtype=np.int64) if ids_value is None else np.asarray(ids_value,dtype=np.int64)
    if ids.shape!=(nc,) or np.unique(ids).size!=nc:raise ValueError('indexToCellID must contain one unique ID per cell')
    width=deriv.shape[2]
    # The reference retains its refusal priority for malformed/unsupported
    # metadata. Every valid closed MPAS class follows the device path.
    if (dtype not in (np.dtype('float32'),np.dtype('float64')) or width<2
            or np.any(owners[:,0]==owners[:,1]) or edges.shape[1]+1>width):
        return build_advection_coefficients(mesh,config_scalar_adv_order=order,
            n_vert_levels=n_vert_levels,source_order_v841=True)
    source=tuple(cp.asarray(np.ascontiguousarray(a)) for a in
                 (counts,neighbors[:,:edges.shape[1]],owners,deriv,dc,dv))
    adv,adv3=cp.empty((ne,width),dtype=dtype),cp.empty((ne,width),dtype=dtype)
    cells=cp.empty((ne,width),dtype=cp.int64);number=cp.empty(ne,dtype=cp.int64)
    failure=cp.zeros(ne,dtype=cp.int32)
    if ne:_kernel(dtype.str,width)(((ne+127)//128,),(128,),
        (*source,adv,adv3,cells,number,failure,np.int32(ne),np.int32(edges.shape[1])))
    if bool(failure.any().get()):
        return build_advection_coefficients(mesh,config_scalar_adv_order=order,
            n_vert_levels=n_vert_levels,source_order_v841=True)
    mask=None
    if n_vert_levels is not None:
        if n_vert_levels<1:raise ValueError('n_vert_levels must be positive')
        mask=np.full((n_vert_levels,ne),int(order!=2),dtype=np.int8)
    return AdvectionCoefficients(cp.asnumpy(adv),cp.asnumpy(adv3),cp.asnumpy(number),
                                 cp.asnumpy(cells),order,mask)
