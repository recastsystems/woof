"""WRF 4.7.1 module_diag_misc.F:229-291, native tendency accumulation.

The producers are the actual post-phy_prep_part2 rates. Their history
records undergo native mass coupling/uncoupling, not a finite difference
of transported prognostics. Only qv_diabatic uses the native MP difference.
GPU checkpoints preserve the amounts and epoch. Native WRF restart files
omit these history-only amounts; their warm-start policy is cold zero.
"""
from __future__ import annotations
import numpy as np

NAMES=('athmpten','aqvmpten','athcuten','aqvcuten','aucuten','avcuten',
       'athshten','aqvshten','aushten','avshten','athblten','aqvblten',
       'aublten','avblten','athratenlw','athratensw')
HISTORY={name:(name.upper(),units,description) for name,units,description in (
 ('athmpten','K ','ACCUMULATED THETA TENDENCY DUE TO MICROPHY SCHEME'),
 ('aqvmpten','kg kg-1 ','ACCUMULATED Q_V TENDENCY DUE TO MICROPHY SCHEME'),
 ('athcuten','K ','ACCUMULATED THETA TENDENCY DUE TO CUMULUS SCHEME'),
 ('aqvcuten','kg kg-1 ','ACCUMULATED Q_V TENDENCY DUE TO CUMULUS SCHEME'),
 ('aucuten','m s-1 ','ACCUMULATED X WIND TENDENCY DUE TO CUMULUS SCHEME'),
 ('avcuten','m s-1 ','ACCUMULATED Y WIND TENDENCY DUE TO CUMULUS SCHEME'),
 ('athshten','K ','ACCUMULATED THETA TENDENCY DUE TO SHALLOW SCHEME'),
 ('aqvshten','kg kg-1 ','ACCUMULATED Q_V TENDENCY DUE TO SHALLOW SCHEME'),
 ('aushten','m s-1 ','ACCUMULATED X WIND TENDENCY DUE TO SHALLOW SCHEME'),
 ('avshten','m s-1 ','ACCUMULATED Y WIND TENDENCY DUE TO SHALLOW SCHEME'),
 ('athblten','K ','ACCUMULATED THETA TENDENCY DUE TO PBL SCHEME'),
 ('aqvblten','kg kg-1 ','ACCUMULATED Q_V TENDENCY DUE TO PBL SCHEME'),
 ('aublten','m s-1 ','ACCUMULATED X WIND TENDENCY DUE TO PBL SCHEME'),
 ('avblten','m s-1 ','ACCUMULATED Y WIND TENDENCY DUE TO PBL SCHEME'),
 ('athratenlw','K ','ACCUMULATED THETA TENDENCY DUE TO LW RADIATION SCHEME'),
 ('athratensw','K ','ACCUMULATED THETA TENDENCY DUE TO SW RADIATION SCHEME'))}
PERSISTENT=(*NAMES,'tendency_epoch')

def drop_inert(values):
 if values.get('acc_phy_tend',0)==0:values.pop('acc_phy_tend',None)

def scratch_shapes(cfg):
 if not getattr(cfg,'acc_phy_tend',0):return {}
 return {**{name:(cfg.nz,cfg.ny,cfg.nx) for name in NAMES},
         'tendency_qv_before':(cfg.nz,cfg.ny,cfg.nx),'tendency_epoch':(2,)}

def validate(cfg):
 if type(cfg.acc_phy_tend) is not int or cfg.acc_phy_tend not in (0,1):
  raise ValueError('acc_phy_tend must be 0 or 1; native Registry allocates its sixteen accumulators only for 1')
 if cfg.acc_phy_tend:
  if cfg.output_interval_s<60 or cfg.output_interval_s%60:
   raise ValueError('acc_phy_tend requires native positive whole-minute history_interval for its FP32 MOD reset')
  if not cfg.moist:
   raise ValueError('acc_phy_tend reads native qv_diabatic; a water-vapor carrier is required')

def prepare_microphysics(state,cfg):
 if getattr(cfg,'acc_phy_tend',0) and cfg.mp_physics and not cfg.no_mp_heating:
  state.existing_scratch('tendency_qv_before')[...]=state.qv

def update(state,cfg):
 if not getattr(cfg,'acc_phy_tend',0):return
 from woof.core.kernels import get_kernel
 p=state.physics;fields=getattr(p,'fields',{});null=np.uint64(0)
 # The common PBL/CU producer records are uncoupled at the completed
 # acoustic mass by finish_history_mass, just as phy_prep_part2 does.
 rates=(state.h_diabatic,None,
        *(fields.get(k) for k in ('rthcuten','rqvcuten','rucuten','rvcuten')),
        None,None,None,None,
        *(fields.get(k) for k in ('rthblten','rqvblten','rublten','rvblten')),
        getattr(p,'rthratenlw',None),getattr(p,'rthratensw',None))
 s=state.existing_scratch;epoch=float(getattr(state,'domain_start_offset',0.) or 0.)
 sz=int(cfg.spec_zone) if cfg.specified or cfg.nested else 0
 get_kernel('output_diagnostics','native_physics_tendency_accumulation')(
  (((cfg.nz*cfg.ny*cfg.nx+255)//256),),(256,),
  (*(null if a is None else a for a in rates),*(s(k) for k in NAMES),
   state.qv,s('tendency_qv_before'),s('tendency_epoch'),
   np.int32(cfg.nz*cfg.ny*cfg.nx),np.int32(cfg.nx),np.int32(cfg.ny),
   np.int32(sz),np.int32(bool(cfg.mp_physics and not cfg.no_mp_heating)),
   np.int32(bool(cfg.cu_physics)),np.int32(0),
   np.float64(state.elapsed_seconds-epoch),np.float32(cfg.output_interval_s),np.float32(cfg.dt)))

def registry_rows(registry):
 registry['parameters']['acc_phy_tend']={'type':'integer','default':0,'enum':[0,1],
  'per_domain':True,'source_pin':'f52c197ed39d12e087d02c50f412d90d418f6186',
  'consuming_read':'woof/core/physics_tendency_output.py'}
 registry.setdefault('diagnostic_state',{})['accumulated_physics_tendencies']={
  name:{'description':d,'units':u,'history':True,'restart':True}
  for name,(_,u,d) in HISTORY.items()}
 registry['diagnostic_state']['accumulated_physics_tendencies']['tendency_epoch']={
  'description':'Native restart-relative tendency window origin (split real64)',
  'units':'s','history':False,'restart':True}
