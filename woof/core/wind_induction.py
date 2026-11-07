"""Native WRF Fitch axial induction correction (Fraunhofer MIT additions)."""
import cupy as cp
import numpy as np
from woof.core.wind_farm import WindFarm
from woof.core.wind_mav import MavFarm
from woof.core.kernels import get_kernel

class InductionFarm(WindFarm):
    def __init__(self,cfg,shape,**kwargs):
        super().__init__(cfg,shape,**kwargs)
        rows=self.data['turbines'];nt=self.nt
        specs=[self.data['types'][r['type']] for r in rows]
        raw=np.zeros((100,nt,3),np.float32)
        for t,s in enumerate(specs):raw[:len(s['native_curve']),t]=s['native_curve']
        self.metadata=(None,None,cp.asarray([r['hub_height_m'] for r in rows],cp.float32),
                       cp.asarray([r['rotor_diameter_m'] for r in rows],cp.float32))
        self.nv=cp.asarray([len(s['native_curve']) for s in specs],cp.int32)
        self.native_curves=tuple(cp.asarray(raw[:,:,i].copy()) for i in range(3))
        self.stc,self.stc2=(cp.asarray([s[key] for s in specs],cp.float32) for key in ('standing_ct','cutout_ct'))
        self.members=cp.ascontiguousarray(cp.argsort(self.col,kind='stable'),cp.int32)
        self.power=cp.zeros(shape[1:],cp.float32);self.thrust=cp.zeros_like(self.power)
        self.tke_increment=cp.zeros(shape,cp.float32);self.turbine_power=cp.zeros(nt,cp.float32)
        self.native_kernel=get_kernel('wind_mav','fitch_induction')
        self.native_block=max(32,1<<(shape[0]-1).bit_length())
        if self.native_block>1024:raise ValueError('Fitch induction supports at most 1024 model mass levels')
        self.native_grid=(int(self.active_columns.size),)
        self.native_prefix=(self.active_columns,self.ptr,self.members,self.metadata[3],self.metadata[2],self.nv,
                            *self.native_curves,self.stc,self.stc2)
        self.native_suffix=(np.int32(self.nc),np.int32(shape[0]),np.int32(nt),self.dx,self.dt,
                            np.float32(self.options['tke_factor']))
    def apply_mynn(self,atmosphere,fields,rates,*,step_seconds=None):
        self.cost.begin();nz=self.shape[0]
        if step_seconds is not None and step_seconds!=self.dt:
            self.dt=np.float32(step_seconds)
            self.native_suffix=(*self.native_suffix[:4],self.dt,self.native_suffix[-1])
        self.native_kernel(self.native_grid,(self.native_block,),
            self.native_prefix+(atmosphere['u'],atmosphere['v'],atmosphere['dz'],atmosphere['z_interface'],
             rates['du'],rates['dv'],fields['qke'],fields.get('qke_adv',fields['qke']),self.power,self.thrust,
             self.tke_rate,self.tke_increment,self.hub,self.turbine_power)+self.native_suffix,shared_mem=nz*4)
        self.cost.end()
    def sample(self,*args):return None
    def history_fields(self):return dict(MavFarm.history_fields(self),THRUST=self.thrust)
    def receipt(self):
        return dict(schema='gpuwm-wind-receipt.v1',scheme='fitch',induction=1,turbine_count=self.nt,
                    table_sha256=self.table_sha256,cost=self.cost.receipt(),turbine_ids=self.ids,farm_ids=self.farm_ids,
                    input_provenance=self.data.get('provenance',{}))
    def write_replay(self,path):return None
