"""Ordered cell reconstruction and mass-flux divergence for moving meshes."""
from functools import lru_cache
import threading
from types import SimpleNamespace

import numpy as np
from .moving_cuda_work import independent_stream

_LOCK = threading.RLock()
OPTIONS = ('--fmad=false',)


@lru_cache(maxsize=2)
def _projection_order(strided=False):
    """Recognize NumPy's three-lane einsum reduction, without assuming its SIMD width."""
    rows=np.array([[1e16,1.,-1e16],[1e16,-1e16,1.],[1.,1e16,-1e16]])
    sample=np.resize(rows,(2,17,3))
    expected=np.einsum('lcj,cj->lc',sample,np.ones((17,3),order='F' if strided else 'C'))
    for order in ((0,1,2),(0,2,1),(1,2,0)):
        actual=(sample[:,:,order[0]]+sample[:,:,order[1]])+sample[:,:,order[2]]
        if actual.tobytes()==expected.tobytes():
            return order
    return None


@lru_cache(maxsize=1)
def _kernels():
    import cupy as cp
    code = r'''
    extern "C" __global__ void reconstruct_mass(
        const long long *edges, const int *counts, const double *coefficients,
        const double *normal, const double *rho, double *out,
        const int nc, const int ne, const int levels, const int width) {
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=(long long)levels*nc*3) return;
        const int component=i%3, cell=(i/3)%nc, level=i/((long long)nc*3);
        double sum=0.;
        for(int slot=0;slot<width && slot<counts[cell];++slot) {
            const long long q=(long long)cell*width+slot;
            sum=__dadd_rn(sum,__dmul_rn(normal[(long long)level*ne+edges[q]],
                                      coefficients[q*3+component]));
        }
        out[i]=__dmul_rn(sum,rho[(long long)level*nc+cell]);
    }
    extern "C" __global__ void divergence(
        const long long *edges, const int *counts, const double *signs,
        const double *length, const double *area, const double *flux, double *out,
        const int nc, const int ne, const int levels, const int width) {
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=(long long)levels*nc) return;
        const int cell=i%nc, level=i/nc;
        double sum=0.;
        for(int slot=0;slot<width && slot<counts[cell];++slot) {
            const long long q=(long long)cell*width+slot, edge=edges[q];
            const double carried=__dmul_rn(__dmul_rn(flux[(long long)level*ne+edge],length[edge]),signs[q]);
            sum=__dadd_rn(sum,carried);
        }
        out[i]=__ddiv_rn(sum,area[cell]);
    }
    extern "C" __global__ void tangent(
        const double *vector,const double *centres,double *out,double *radial,
        const int nc,const long long count){
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=count)return;
        const int cell=i%nc;
        double r=0.;
        const int order[3]={ORDER};
        for(int slot=0;slot<3;++slot){const int j=order[slot];
            r=__dadd_rn(r,__dmul_rn(vector[3*i+j],centres[3*cell+j]));}
        radial[i]=r;
        for(int j=0;j<3;++j)out[3*i+j]=__dsub_rn(vector[3*i+j],__dmul_rn(r,centres[3*cell+j]));
    }
    extern "C" __global__ void edge_projection(
        const double *vector,const long long *cells,const double *normals,
        float *out,const int nc,const int ne,const long long count,const int o0,const int o1,const int o2){
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=count)return;
        const int edge=i%ne,level=i/ne;
        const long long a=cells[2*edge],b=cells[2*edge+1];
        double sum=0.;
        const int order[3]={o0,o1,o2};
        for(int slot=0;slot<3;++slot){const int j=order[slot];
            const double va=a<0?0.:vector[((long long)level*nc+a)*3+j];
            const double vb=b<0?0.:vector[((long long)level*nc+b)*3+j];
            const double mean=__dmul_rn(__dadd_rn(va,vb),.5);
            sum=__dadd_rn(sum,__dmul_rn(mean,normals[3*edge+j]));
        }
        float rounded;
        asm("cvt.rn.f32.f64 %0,%1;" : "=f"(rounded) : "d"(sum));
        out[i]=rounded;
    }
    '''
    code=code.replace('ORDER',','.join(str(i) for i in (_projection_order() or (0,1,2))))
    return tuple(cp.RawKernel(code, name, options=OPTIONS) for name in
                 ('reconstruct_mass','divergence','tangent','edge_projection'))


def _geometry(weights, arrays, kind):
    import cupy as cp
    with _LOCK:
        entries = getattr(weights, '_cuda_momentum_geometry', None)
        if entries is None:
            entries = weights._cuda_momentum_geometry = {}
        device = cp.cuda.runtime.getDevice()
        old = entries.get(kind)
        if old is not None and old['source'] is arrays and old['device'] == device:
            return old
        from .acoustic import edge_signs_on_cells
        edges = np.asarray(arrays['edgesOnCell'], dtype=np.int64)
        counts = np.asarray(arrays['nEdgesOnCell'], dtype=np.int32)
        ne = len(arrays['dvEdge'])
        active = np.arange(edges.shape[1])[None, :] < counts[:, None]
        if np.any(active & ((edges < 0) | (edges >= ne))):
            raise ValueError('moving momentum has invalid active edge')
        coeff = np.asarray(arrays.get('coeffs_reconstruct', np.empty(0)), dtype=np.float64)
        raw = (edges, counts, edge_signs_on_cells(SimpleNamespace(arrays=arrays)).astype(np.float64),
               np.asarray(arrays['dvEdge'], dtype=np.float64),
               np.asarray(arrays['areaCell'], dtype=np.float64), coeff)
        entry = dict(source=arrays, device=device,
                     arrays=tuple(cp.asarray(np.ascontiguousarray(v)) for v in raw))
        cp.cuda.get_current_stream().synchronize()
        entries[kind] = entry
        return entry


@independent_stream
def prepare(weights, source, target):
    _geometry(weights, source, 'source')
    _geometry(weights, target, 'target')
    if _projection_order() is not None and all(n in target for n in ('xCell','yCell','zCell','edgeNormalVectors')):
        _projection_geometry(weights, target)


def _projection_geometry(weights, arrays):
    import cupy as cp
    with _LOCK:
        old = getattr(weights, '_cuda_projection_geometry', None)
        device = cp.cuda.runtime.getDevice()
        if old is not None and old['source'] is arrays and old['device'] == device:
            return old
        centres = np.stack([arrays[n] for n in ('xCell','yCell','zCell')], axis=-1).astype(np.float64)
        centres /= np.linalg.norm(centres, axis=-1, keepdims=True)
        cells = np.asarray(arrays['cellsOnEdge'], dtype=np.int64)
        normals = np.asarray(arrays['edgeNormalVectors'], dtype=np.float64)
        if cells.ndim != 2 or cells.shape[1] != 2 or np.any(cells >= len(centres)):
            raise ValueError('moving edge projection has invalid cell references')
        if normals.shape != (len(cells), 3):
            raise ValueError(f'edge normal shape {normals.shape} is unsupported')
        entry = dict(source=arrays, device=device, cells=cells,
                     gather_cells=np.where(cells < 0, len(centres), cells),
                     order=_projection_order(normals.strides[-1] != normals.dtype.itemsize),
                     arrays=tuple(cp.asarray(np.ascontiguousarray(x)) for x in (centres,cells,normals)))
        cp.cuda.get_current_stream().synchronize()
        weights._cuda_projection_geometry = entry
        return entry


@independent_stream
def project_edges(weights, vector, arrays):
    import cupy as cp
    if _projection_order() is None:
        return None
    geo = _projection_geometry(weights, arrays)
    if geo['order'] is None:
        return None
    centres, cells, normals = geo['arrays']
    source = cp.asarray(np.ascontiguousarray(vector, dtype=np.float64))
    nl, nc, _ = source.shape
    ne = len(cells)
    out, radial = cp.empty_like(source), cp.empty((nl,nc), dtype=cp.float64)
    flux = cp.empty((nl,ne), dtype=cp.float32)
    _kernels()[2](((radial.size+127)//128,), (128,),
                  (source, centres, out, radial, np.int32(nc), np.int64(radial.size)))
    _kernels()[3](((flux.size+127)//128,), (128,),
                  (out, cells, normals, flux, np.int32(nc), np.int32(ne), np.int64(flux.size),
                   *(np.int32(i) for i in geo['order'])))
    return cp.asnumpy(out), cp.asnumpy(radial), geo['cells'], geo['gather_cells'], cp.asnumpy(flux)


@independent_stream
def reconstruct_mass(weights, normal, rho, arrays):
    import cupy as cp
    geo = _geometry(weights, arrays, 'source')['arrays']
    edges, counts, _, _, _, coeff = geo
    nc, width = edges.shape
    if coeff.shape != (nc, width, 3):
        raise ValueError(f'coeffs_reconstruct layout {coeff.shape} != {(nc, width, 3)}')
    normal, rho = (cp.asarray(np.ascontiguousarray(v, dtype=np.float64)) for v in (normal, rho))
    out = cp.empty((normal.shape[0], nc, 3), dtype=cp.float64)
    _kernels()[0](((out.size+127)//128,), (128,), (edges, counts, coeff, normal, rho, out,
        np.int32(nc), np.int32(normal.shape[1]), np.int32(normal.shape[0]), np.int32(width)))
    return cp.asnumpy(out)


@independent_stream
def divergence(weights, flux, arrays, *, kind):
    import cupy as cp
    edges, counts, signs, length, area, _ = _geometry(weights, arrays, kind)['arrays']
    nc, width = edges.shape
    flux = cp.asarray(np.ascontiguousarray(flux, dtype=np.float64))
    out = cp.empty((flux.shape[0], nc), dtype=cp.float64)
    _kernels()[1](((out.size+127)//128,), (128,), (edges, counts, signs, length, area, flux, out,
        np.int32(nc), np.int32(flux.shape[1]), np.int32(flux.shape[0]), np.int32(width)))
    return cp.asnumpy(out)
