"""Native WRF 4.7.1 water/energy carry counters (public-domain pin).

module_diag_misc.F:296-431: one subtraction, strict greater-than, inside
bucket_mm's hourly gate. Counter arrays are INTEGER, including on disk.
"""
from __future__ import annotations
import numpy as np
from woof.core.physics_inventory import RADIATION_BOUNDARY_FLUXES
AMOUNTS = ('rainc', 'rainnc', *('ac'+name for name in RADIATION_BOUNDARY_FLUXES))
COUNTERS = tuple('i_'+name for name in AMOUNTS)
HISTORY = {name: (name.upper(), '', 'BUCKET FOR '+amount.upper())
           for name,amount in zip(COUNTERS,AMOUNTS)}
PERSISTENT = (*COUNTERS, 'bucket_epoch')

def drop_inert(values):
    for name in ('bucket_mm','bucket_j'):
        if values.get(name,-1.)<=0:values.pop(name,None)

def scratch_shapes(cfg):
    names = COUNTERS[:2] if cfg.bucket_mm>0 else ()
    if cfg.bucket_j>0:names += COUNTERS[2:]
    result={name:(cfg.ny,cfg.nx) for name in names}
    if cfg.bucket_mm>0:
        result['bucket_epoch']=(2,)
        result['output_zero']=(cfg.ny,cfg.nx)
    return result

def update(state,cfg):
    if cfg.bucket_mm<=0:return
    from woof.core.kernels import get_kernel
    physics=state.physics;fields=getattr(physics,'fields',{})
    zero=state.existing_scratch('output_zero')
    micro=getattr(physics,'microphysics',None)
    arrays=(getattr(physics,'rainc',None),getattr(micro,'rainnc',None),
            *(fields.get(name) for name in AMOUNTS[2:]))
    epoch=float(getattr(state,'domain_start_offset',0.) or 0.)
    counters=tuple(state.existing_scratch(name) for name in COUNTERS)
    null=np.uint64(0)
    get_kernel('output_diagnostics','native_accumulation_buckets')(
        (((cfg.nx*cfg.ny+255)//256),),(256,),
        (*(null if array is None else array for array in arrays),
         *(null if array is None else array for array in counters),
         state.existing_scratch('bucket_epoch'),np.int32(cfg.nx*cfg.ny),
         np.float64(state.elapsed_seconds-epoch),
         np.float32(state.elapsed_seconds/60.),np.float32(cfg.bucket_mm),
         np.float32(cfg.bucket_j)))

def registry_rows(registry):
    for name,units in (('bucket_mm','mm'),('bucket_j','J m-2')):
        registry['parameters'][name]={'type':'real','default':-1.,'per_domain':False,
            'units':units,'source_pin':'f52c197ed39d12e087d02c50f412d90d418f6186',
            'consuming_read':'woof/core/misc_buckets.py'}
    registry.setdefault('diagnostic_state',{})['accumulation_buckets']={
        name:{'description':d,'units':u,'dtype':'int32','history':True,'restart':True}
        for name,(_,u,d) in HISTORY.items()}
    registry['diagnostic_state']['accumulation_buckets']['bucket_epoch']={
        'description':'Native restart-relative hourly clock origin (split real64)',
        'units':'s','history':False,'restart':True}
