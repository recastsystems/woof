"""Stock public WRF aerosol number optics, isolated from the earlier fork.

The native pressure above the last mass level is undefined in phy_prep.
The caller supplies an explicit halo; production uses a defined zero word.
The captured native halo remains part of the module replay input.
"""
from __future__ import annotations
import numpy as np


def optics_device(p, t, qv, dz, nwfa, nifa, *, pressure_halo=None):
    import cupy as cp
    from woof.wrf_exact import ENABLED
    from woof.core.kernels import get_kernel
    nc,nz=p.shape
    arrays=tuple(cp.ascontiguousarray(a,dtype=cp.float32) for a in (p,t,qv,dz,nwfa,nifa))
    if nz<2 or any(a.shape!=(nc,nz) for a in arrays):
        raise ValueError('stock WRF aerosol optics require complete (column,level) profiles')
    halo=(cp.zeros(nc,cp.float32) if ENABLED else arrays[0]) if pressure_halo is None else cp.ascontiguousarray(pressure_halo,dtype=cp.float32)
    if pressure_halo is not None and halo.shape!=(nc,):
        raise ValueError('stock WRF aerosol pressure halo must have one value per column')
    nl=nz+1
    tau=cp.empty((nc,14,nl),cp.float32);asy=cp.empty_like(tau);ssa=cp.empty_like(tau)
    aod=cp.empty((nc,nz),cp.float32)
    get_kernel('wrf_stock_aer3','wrf_stock_aer3_optics')(
        ((nc*nl+127)//128,),(128,),
        (np.int32(nc),np.int32(nz),np.int32(nl),*arrays,tau,asy,ssa,aod))
    # Diagnostics use the native model layers, without the SW buffer layer.
    diagnostics=cp.empty((4,nc),cp.float32)
    if ENABLED:
        profiles=tuple(cp.ascontiguousarray(a[:,:,:nz]) for a in (tau,asy,ssa))
        pressure=cp.concatenate((arrays[0],halo[:,None]),axis=1)
        get_kernel('wrf_stock_aer3','wrf_stock_aer3_diagnostics')(
            ((nc+127)//128,),(128,),
            (np.int32(nc),np.int32(nz),pressure,*profiles,aod,diagnostics))
    else:
        get_kernel('wrf_stock_aer3','wrf_stock_aer3_diagnostics_fast')(
            ((nc*32+127)//128,),(128,),
            (np.int32(nc),np.int32(nz),np.int32(pressure_halo is not None),arrays[0],halo,tau,asy,ssa,aod,diagnostics))
    # AOD5503D is the user-aerosol field; native aer_opt=3 leaves it alone.
    return (tau,asy,ssa),aod,diagnostics
