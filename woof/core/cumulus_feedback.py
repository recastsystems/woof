"""The shared subgrid cloud handoff to radiation, in WRF operation order."""
from __future__ import annotations
import numpy as np


def merge_cumulus_feedback(qc,qi,qs,cldfra,feedback,*,aerosol=False,radii=None,radii_meters=False):
    import cupy as cp
    from woof.core.kernels.no_ftz import get_kernel_no_ftz as get_kernel
    shape=qc.shape
    for name in ('cldfra_dp','cldfra_sh','qc_cu','qi_cu','qs_cu','efcs','efis','efss'):
        if feedback[name].shape!=shape or feedback[name].dtype!=cp.float32:
            raise ValueError('cumulus-radiation field '+name+' must have the radiation profile shape; a different grid would mix unrelated columns')
    if aerosol and (radii is None or any(radii.get(n) is None for n in ('effc','effi','effs'))):
        raise ValueError('aerosol cumulus-radiation coupling needs all three resolved effective radii before their weighted blend')
    qc=cp.ascontiguousarray(qc).copy()
    qi=cp.ascontiguousarray(qi).copy()
    qs=cp.ascontiguousarray(qs).copy()
    cldfra=cp.ascontiguousarray(cldfra).copy()
    output_radii=None if radii is None else dict(radii)
    if aerosol:
        for n in ('effc','effi','effs'):
            output_radii[n]=cp.ascontiguousarray(radii[n]).copy()
        ec,ei,es=(output_radii[n] for n in ('effc','effi','effs'))
    else:
        ec=ei=es=qc
    n=qc.size
    values=tuple(cp.ascontiguousarray(feedback[n]) for n in ('cldfra_dp','cldfra_sh','qc_cu','qi_cu','qs_cu','efcs','efis','efss'))
    get_kernel('cumulus_feedback','cumulus_feedback')((int((n+255)//256),),(256,),
        (qc,qi,qs,cldfra)+values+(ec,ei,es,np.int32(aerosol),np.int32(radii_meters),np.int64(n)))
    return qc,qi,qs,cldfra,output_radii
