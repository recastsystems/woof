"""Staged CAM3 LW path kernel, not a radiation scheme admission."""
from __future__ import annotations

import numpy as np

PATH_FIELDS = ('plco2','plh2o','tplnka','s2c','tcg','w','tplnke',
               'tint','tint4','tlayr','tlayr4','plh2ob','wb')


def lw_paths(tnm, qnm, pint, pmln, piln, lwupcgs, constants):
    """Evaluate WRF radtpl on binary64 top-to-bottom column batches."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    inputs=[cp.ascontiguousarray(cp.asarray(x,dtype=cp.float64))
            for x in (tnm,qnm,pint,pmln,piln,lwupcgs,constants)]
    t,q,p,ml,il,sfc,const=inputs
    if t.ndim!=2 or q.shape!=t.shape or ml.shape!=t.shape:
        raise ValueError('CAM3 paths require matching (column,level) T, q and log pressure')
    ncol,pver=t.shape
    if ncol<1 or pver<1 or p.shape!=(ncol,pver+1) or il.shape!=p.shape:
        raise ValueError('CAM3 interface pressures require one more interface than layers')
    if sfc.shape!=(ncol,) or const.shape!=(10,):
        raise ValueError('CAM3 paths require a surface flux per column and ten initialized constants')
    output=cp.zeros((ncol,15,pver+1),dtype=cp.float64)
    get_kernel('cam3_paths','cam_lw_paths')(
        ((ncol+63)//64,), (64,),
        (np.int32(ncol),np.int32(pver),*inputs,output))
    result={name:output[:,i,:] for i,name in enumerate(PATH_FIELDS[:11])}
    result['tplnke']=output[:,6,0:1]
    result['plh2ob']=output[:,11:13,:]
    result['wb']=output[:,13:15,:]
    return result
