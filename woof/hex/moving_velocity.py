"""Terrain-coupled W recovery preserving the scalar authority's FP order."""
import numpy as np
from woof.hex.acoustic import edge_signs_on_cells


def recover_vertical_velocity(state, vertical, arrays, terrain, *, device_map=None):
    rho, ru, rw = state.rho, state.rho_u, state.rho_w
    nlev, ncells = rho.shape
    dtype = rho.dtype
    fzm, fzp, zz = vertical.fzm, vertical.fzp, vertical.zz
    cf1, cf2, cf3 = (dtype.type(getattr(vertical, name)) for name in ('cf1','cf2','cf3'))
    one = dtype.type(1.)
    zb=np.asarray(terrain.zb_cell,dtype=dtype)
    zb3=np.asarray(terrain.zb3_cell,dtype=dtype)
    if nlev < 3 or not np.isfinite([cf1,cf2,cf3]).all():
        raise ValueError('W recovery requires three levels and finite bottom coefficients')
    if np.any(rho <= 0) or np.any(zz <= 0):
        raise ValueError('W recovery requires positive density and vertical metric')
    import os
    if (getattr(device_map, 'backend', 'cpu') == 'cuda'
            and os.environ.get('WOOF_HEX_MOVING_W_RECOVERY_GPU', '1') != '0'
            and dtype in (np.dtype('float32'), np.dtype('float64'))
            and all(np.asarray(a).dtype == dtype for a in (ru, rw, zz, fzm, fzp))):
        from .moving_velocity_cuda import recover
        return recover(device_map, state, vertical, arrays, terrain)
    from types import SimpleNamespace
    signs = edge_signs_on_cells(SimpleNamespace(arrays=arrays)).astype(dtype, copy=False)
    counts, edges = arrays['nEdgesOnCell'], arrays['edgesOnCell']
    w = np.zeros((nlev+1,ncells),dtype=dtype)
    metric = fzm[1:nlev,None]*zz[1:] + fzp[1:nlev,None]*zz[:-1]
    w[1:nlev] = rw[1:nlev]/metric
    # Keep each cell's slot accumulation order, including the cf bottom
    # diagnosis. Vectorize only independent cells and levels. No reduction
    # tree, fused expression or precision change replaces the scalar order.
    for slot in range(edges.shape[1]):
        ids = np.flatnonzero(counts > slot)
        selected = edges[ids,slot]
        if np.any(selected < 0) or np.any(selected >= ru.shape[1]):
            raise ValueError('W recovery has invalid active edge')
        flux = cf1*ru[0,selected] + cf2*ru[1,selected] + cf3*ru[2,selected]
        w[0,ids] = w[0,ids] + signs[ids,slot]*(zb[0,ids,slot] +
                    np.copysign(one,flux)*zb3[0,ids,slot])*flux
        flux = fzm[1:nlev,None]*ru[1:,selected] + fzp[1:nlev,None]*ru[:-1,selected]
        w[1:nlev,ids] = w[1:nlev,ids] + signs[ids,slot]*(zb[1:nlev,ids,slot] +
                    np.copysign(one,flux)*zb3[1:nlev,ids,slot])*flux
    bottom = cf1*rho[0] + cf2*rho[1] + cf3*rho[2]
    density = fzm[1:nlev,None]*rho[1:] + fzp[1:nlev,None]*rho[:-1]
    if np.any(bottom == 0) or np.any(density == 0):
        raise FloatingPointError('zero W recovery density denominator')
    w[0] /= bottom
    w[1:nlev] /= density
    return w
