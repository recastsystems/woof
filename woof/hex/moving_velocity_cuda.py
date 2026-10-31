"""Terrain W recovery on the card with the host's slot and rounding order."""
from functools import lru_cache
import threading
import numpy as np
from .moving_cuda_work import independent_stream

_LOCK = threading.RLock()
OPTIONS = ('--fmad=false', '--ftz=false')


@lru_cache(maxsize=2)
def _kernel(dtype):
    import cupy as cp
    single = np.dtype(dtype) == np.dtype('float32')
    real, constraint, suffix = ('float', 'f', 'f32') if single else ('double', 'd', 'f64')
    # Inline PTX omits .ftz explicitly. A moving remap must retain the host's
    # subnormal words even when the forecast's fast numerics enable FTZ.
    helpers = '\n'.join(
        f'__device__ __forceinline__ {real} {name}({real} a,{real} b) {{ {real} r; '
        f'asm("{opcode}.rn.{suffix} %0,%1,%2;" : "={constraint}"(r) : "{constraint}"(a),"{constraint}"(b)); return r; }}'
        for name, opcode in [('A', 'add'), ('M', 'mul'), ('D', 'div')])
    code = helpers + r'''
    extern "C" __global__ void recover(
        const long long *edges, const int *counts, const REAL *signs,
        const REAL *zz, const REAL *fzm, const REAL *fzp,
        const REAL *zb, const REAL *zb3, const REAL *rho, const REAL *ru,
        const REAL *rw, REAL *w, int *invalid, const REAL cf1,
        const REAL cf2, const REAL cf3, const int nc, const int ne,
        const int nl, const int width) {
        const long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(i>=(long long)(nl+1)*nc) return;
        const int cell=i%nc, level=i/nc;
        if(level==nl) {w[i]=0;return;}
        REAL result=0, density;
        if(level==0) density=A(A(M(cf1,rho[cell]),M(cf2,rho[nc+cell])),M(cf3,rho[2*nc+cell]));
        else {
            density=A(M(fzm[level],rho[i]),M(fzp[level],rho[i-nc]));
            const REAL metric=A(M(fzm[level],zz[i]),M(fzp[level],zz[i-nc]));
            result=D(rw[i],metric);
        }
        for(int slot=0;slot<width && slot<counts[cell];++slot) {
            const long long q=(long long)cell*width+slot, edge=edges[q];
            REAL flux;
            if(level==0) flux=A(A(M(cf1,ru[edge]),M(cf2,ru[ne+edge])),M(cf3,ru[2*ne+edge]));
            else flux=A(M(fzm[level],ru[(long long)level*ne+edge]),
                        M(fzp[level],ru[(long long)(level-1)*ne+edge]));
            const long long t=i*width+slot;
            const REAL slope=A(zb[t],M(copysign((REAL)1,flux),zb3[t]));
            result=A(result,M(M(signs[q],slope),flux));
        }
        if(density==0) atomicExch(invalid,1);
        w[i]=D(result,density);
    }
    '''.replace('REAL', real)
    return cp.RawKernel(code, 'recover', options=OPTIONS)


def _geometry(weights, vertical, arrays, terrain):
    import cupy as cp
    with _LOCK:
        old = getattr(weights, '_cuda_velocity_geometry', None)
        identity = (id(vertical), id(arrays), id(terrain), cp.cuda.runtime.getDevice())
        if old is not None and old['identity'] == identity:
            return old
        from .acoustic import edge_signs_on_cells
        from types import SimpleNamespace
        dtype = vertical.zz.dtype
        edges = np.asarray(arrays['edgesOnCell'], dtype=np.int64)
        counts = np.asarray(arrays['nEdgesOnCell'], dtype=np.int32)
        active = np.arange(edges.shape[1])[None, :] < counts[:, None]
        ne = len(arrays['cellsOnEdge'])
        if np.any(active & ((edges < 0) | (edges >= ne))):
            raise ValueError('W recovery has invalid active edge')
        fields = (edges, counts, edge_signs_on_cells(SimpleNamespace(arrays=arrays)).astype(dtype),
                  vertical.zz, vertical.fzm, vertical.fzp,
                  np.asarray(terrain.zb_cell, dtype=dtype), np.asarray(terrain.zb3_cell, dtype=dtype))
        out = dict(identity=identity, owners=(vertical, arrays, terrain),
                   arrays=tuple(cp.asarray(np.ascontiguousarray(a)) for a in fields))
        cp.cuda.get_current_stream().synchronize()
        weights._cuda_velocity_geometry = out
        return out


@independent_stream
def prepare(weights, vertical, arrays, terrain):
    import os
    import cupy as cp
    from .class_work_arena import _ARENAS
    live = cp.get_default_memory_pool().used_bytes()+sum(a.large.used_bytes() for a in _ARENAS)
    estimate = sum(np.asarray(a).nbytes for a in (vertical.zz, terrain.zb_cell, terrain.zb3_cell))+(32 << 20)
    limit = int(os.environ.get('WOOF_HEX_MOVING_PREPARE_BUDGET_MIB', '20000'))*2**20
    reserve = int(os.environ.get('WOOF_HEX_MOVING_TRANSIENT_MIB', '4096'))*2**20
    if live+estimate+reserve > limit:
        return False
    _geometry(weights, vertical, arrays, terrain)
    return True


@independent_stream
def recover(weights, state, vertical, arrays, terrain):
    import cupy as cp
    dtype = state.rho.dtype
    geo = _geometry(weights, vertical, arrays, terrain)['arrays']
    rho, ru, rw = (cp.asarray(np.ascontiguousarray(v)) for v in (state.rho, state.rho_u, state.rho_w))
    nl, nc = rho.shape
    out, invalid = cp.empty((nl+1, nc), dtype=dtype), cp.zeros(1, dtype=cp.int32)
    _kernel(dtype.str)(((out.size+127)//128,), (128,), (*geo, rho, ru, rw, out, invalid,
        *(dtype.type(getattr(vertical, n)) for n in ('cf1', 'cf2', 'cf3')),
        np.int32(nc), np.int32(ru.shape[1]), np.int32(nl), np.int32(geo[0].shape[1])))
    result = cp.asnumpy(out)
    if int(invalid.get()[0]):
        raise FloatingPointError('zero W recovery density denominator')
    return result
