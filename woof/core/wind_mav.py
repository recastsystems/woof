"""Resident WRF MAV graph, rotor and column coupling.

The graph is rebuilt on device from the current winds every call. No wake
or rotor workspace crosses a model step; prognostic QKE uses the existing
MYNN state, nest exchange and checkpoint carriers.
"""
from __future__ import annotations
import numpy as np
import cupy as cp
from woof.core.kernels import get_kernel

ENSEMBLES = {1: ((1, 0),), 2: ((2, 0),), 3: ((3, 0),),
             4: ((1, 4), (2, 3)), 5: ((1, 4), (2, 3), (3, 2)),
             6: ((1, 3), (1, 4), (2, 3), (3, 2)),
             7: ((1, 4), (2, 3), (2, 4), (3, 2))}


class MavWorkspace:
    def __init__(self, shape, col, xt, yt, hub, diameter, area, nv, ws, tc, pc, stc, stc2,
                 *, dx, dt, correction=.25, model=2, overlap=4, tkeadvect=True):
        self.shape = tuple(shape)
        nz, ny, nx = self.shape
        self.nc, self.nt = ny*nx, int(col.size)
        self.col = cp.asarray(col, dtype=cp.int32)
        nt = self.nt
        self.metadata = tuple(cp.ascontiguousarray(cp.asarray(a, dtype=dtype)) for a, dtype in
                              [(xt,cp.float32),(yt,cp.float32),(hub,cp.float32),(diameter,cp.float32),
                               (area,cp.float32),(self.col,cp.int32),(nv,cp.int32),(ws,cp.float32),
                               (tc,cp.float32),(pc,cp.float32),(stc,cp.float32),(stc2,cp.float32)])
        plan = ENSEMBLES[model]
        self.models, self.arms = len(plan), 7*len(plan)
        directions = np.asarray([-2.5,-1.5,-.5,0.,.5,1.5,2.5], np.float32)
        # Native float32 EXP is evaluated by the shared device math, once.
        self.weights = cp.empty(self.arms,cp.float32)
        self.directions = cp.empty(self.arms,cp.float32)
        get_kernel('wind_mav','mav_prepare_arms')((1,),(32,),
            (self.weights,self.directions,np.int32(self.models)))
        self.model = cp.asarray([m for _ in range(7) for m,_ in plan],cp.int32)
        self.method = cp.asarray([o or overlap for _ in range(7) for _,o in plan],cp.int32)
        self.ui,self.ul = (cp.zeros((self.arms,nt),cp.float32) for _ in range(2))
        self.work = cp.empty(self.arms*(5*nt+6*nt*nt),cp.float32)
        self.iwork = cp.empty(self.arms*(3*nt+nt*nt),cp.int32)
        self.status = cp.zeros(self.arms,cp.int32)
        self.rotor = cp.empty((self.arms,nt,nz,3),cp.float32)
        self.powers = cp.empty((2,self.arms,nt),cp.float32)
        valid = (self.col>=0)&(self.col<self.nc)
        self.members = cp.ascontiguousarray(cp.flatnonzero(valid)[cp.argsort(self.col[valid],kind='stable')],cp.int32)
        counts = cp.bincount(self.col[valid],minlength=self.nc)
        self.ptr = cp.concatenate((cp.zeros(1,cp.int32),cp.cumsum(counts,dtype=cp.int32)))
        self.active_columns = cp.ascontiguousarray(cp.flatnonzero(counts),cp.int32)
        self.hub = cp.zeros((nt,10),cp.float32)
        self.turbine_power = cp.zeros(nt,cp.float32)
        self.power = cp.zeros((ny,nx),cp.float32)
        self.tke_rate = cp.zeros(shape,cp.float32)
        self.tke_increment = cp.zeros(shape,cp.float32)
        self.zm = cp.empty(shape,cp.float32)
        self.dx,self.dt,self.correction = map(np.float32,(dx,dt,correction))
        self.tkeadvect = int(tkeadvect)
        self.kernels = {name:get_kernel('wind_mav',name) for name in ('mav_wake','mav_rotor','mav_accumulate','mav_mass_height')}
        from woof.wrf_exact import ENABLED
        self.strict = ENABLED
        self.accum_grid = (int(self.active_columns.size),) if ENABLED else (int(self.active_columns.size), nz)
        self.block = max(32,1<<(nz-1).bit_length())
        self.accum_block = self.block if ENABLED else 128
        if self.block>1024:
            raise ValueError('MAV rotor vertical grid exceeds the CUDA block limit of 1024 mass levels')

    def apply(self, atmosphere, land, co, si, du, dv, qke, adv, *, step_seconds=None, z_mass=None):
        nz,ny,nx=self.shape
        if step_seconds is not None:self.dt=np.float32(step_seconds)
        if z_mass is None and self.strict:
            self.kernels['mav_mass_height'](((nz*self.nc+127)//128,),(128,),
                (atmosphere['z_interface'],self.zm,np.int32(self.nc),np.int32(nz)))
        if z_mass is None:z_mass=self.zm
        self.kernels['mav_wake']((self.arms,),(128,),
            (atmosphere['u'],atmosphere['v'],atmosphere['z_interface'],z_mass,land,co,si,
             *self.metadata,self.model,self.method,self.directions,self.ui,self.ul,self.work,self.iwork,
             self.status,np.int32(self.nc),np.int32(nz),np.int32(self.nt),self.correction))
        xt,yt,hub,diameter,area,col,nv,ws,tc,pc,stc,stc2=self.metadata
        self.kernels['mav_rotor']((self.nt,self.arms),(self.block,),
            (atmosphere['u'],atmosphere['v'],atmosphere['dz'],atmosphere['z_interface'],self.ui,self.ul,
             diameter,hub,area,col,nv,ws,tc,pc,stc,stc2,self.weights,self.rotor,self.powers,self.status,
             np.int32(self.nc),np.int32(nz),np.int32(self.nt),self.dt,np.float32(1/(self.dx*self.dx)),self.correction),
            shared_mem=nz*4)
        self.kernels['mav_accumulate'](self.accum_grid,(self.accum_block,),
            (self.active_columns,self.ptr,self.members,self.rotor,self.powers,du,dv,qke,adv,
             self.tke_rate,self.tke_increment,self.power,self.turbine_power,self.ul,self.ui,self.weights,self.hub,
             np.int32(self.nc),np.int32(nz),np.int32(self.nt),np.int32(self.arms),np.int32(self.models),
             np.int32(self.tkeadvect),self.dt))


class MavFarm(MavWorkspace):
    def __init__(self,cfg,shape,*,prepared,latitude=None,longitude=None,msf=None,z_interface=None,projection=None):
        from woof.core.wind_farm import WindCost
        data,digest=prepared
        rows=data['turbines'];nt=len(rows);nz,ny,nx=shape
        if not nt:raise ValueError('MAV needs at least one turbine')
        xy=cp.asarray([[r['mav_x'],r['mav_y']] for r in rows],cp.float32)
        xt,yt=(cp.empty(nt,cp.float32) for _ in range(2));col=cp.empty(nt,cp.int32)
        if cfg.windfarm_ij==1:
            get_kernel('wind_mav','mav_layout_xy')(((nt+127)//128,),(128,),
                (xy,xt,yt,col,np.int32(nt),np.int32(nx),np.int32(ny),np.float32(cfg.dx),np.float32(cfg.windfarm_deg)))
        else:
            xt[:]=xy[:,0];yt[:]=xy[:,1]
            ll=cp.asarray([[r['latitude'],r['longitude']] for r in rows],cp.float32)
            if projection is None:
                raise ValueError('Native MAV geographic input requires the domain map projection')
            get_kernel('wind_mav','mav_project_ll')(((nt+127)//128,),(128,),
                (ll,cp.ascontiguousarray(latitude,dtype=cp.float32),cp.ascontiguousarray(longitude,dtype=cp.float32),col,np.int32(nt),np.int32(nx),np.int32(ny),
                 np.int32(cfg.map_proj),np.float32(cfg.dx),np.float32(projection.truelat1),
                 np.float32(projection.truelat2),np.float32(projection.stand_lon)))
        specs=[data['types'][r['type']] for r in rows]
        hub=cp.asarray([r['hub_height_m'] for r in rows],cp.float32)
        diameter=cp.asarray([r['rotor_diameter_m'] for r in rows],cp.float32)
        area=np.float32(np.pi)/np.float32(4)*(diameter*diameter)
        nv=cp.asarray([len(s['mav_curve']) for s in specs],cp.int32)
        curves=np.zeros((100,nt,3),np.float32)
        for t,s in enumerate(specs):curves[:len(s['mav_curve']),t]=s['mav_curve']
        super().__init__(shape,col,xt,yt,hub,diameter,area,nv,
            *[cp.asarray(curves[:,:,i].copy()) for i in range(3)],
            cp.asarray([s['standing_ct'] for s in specs],cp.float32),
            cp.asarray([s['cutout_ct'] for s in specs],cp.float32),dx=cfg.dx,dt=cfg.dt,
            correction=cfg.windfarm_tke_factor,model=cfg.windfarm_wake_model,
            overlap=cfg.windfarm_overlap_method,tkeadvect=cfg.bl_mynn_tkeadvect)
        self.data,self.table_sha256,self.scheme=data,digest,'mav'
        self.ids=[r['id'] for r in rows];self.farm_ids=list(dict.fromkeys(r['farm_id'] for r in rows))
        self.farm_index=cp.asarray([self.farm_ids.index(r['farm_id']) for r in rows],cp.int32)
        self.cost=WindCost();self.replay=None
        self.options=dict(wake_model=cfg.windfarm_wake_model,overlap_method=cfg.windfarm_overlap_method,
                          tke_factor=cfg.windfarm_tke_factor,coordinate_mode=cfg.windfarm_ij)
        self.co=cp.ones((ny,nx),cp.float32);self.si=cp.zeros((ny,nx),cp.float32)

    def apply_mynn(self,atmosphere,fields,rates,*,step_seconds=None,cosa=None,sina=None):
        self.cost.begin()
        self.apply(atmosphere,fields['xland'],self.co if cosa is None else cosa,
                   self.si if sina is None else sina,rates['du'],rates['dv'],fields['qke'],
                   fields.get('qke_adv',fields['qke']),step_seconds=step_seconds)
        self.cost.end()

    def sample(self,atmosphere,fields):
        # Native real.exe initializes POWER to zero before the first physics call.
        # Checkpoint restoration subsequently fills the aliased diagnostic arrays.
        return None

    def history_fields(self):
        wind=cp.ascontiguousarray(self.hub[:,0]);power=self.turbine_power
        count=cp.bincount(self.farm_index,minlength=len(self.farm_ids))
        return {'POWER':self.power,'TKE_WINDFARM':self.tke_increment,'WIND_TKE_SOURCE':self.tke_rate,
            'WIND_TKE_SOURCE_MAX':cp.max(self.tke_rate,axis=0),
            'TURBINE_POWER':power,'TURBINE_HUB_WIND':wind,
            'TURBINE_FARM_INDEX':(self.farm_index+1).astype(cp.float32),
            'TURBINE_HUB_HEIGHT':self.metadata[2],
            'FARM_POWER':cp.bincount(self.farm_index,weights=power,minlength=len(self.farm_ids)).astype(cp.float32),
            'FARM_HUB_WIND':(cp.bincount(self.farm_index,weights=wind,minlength=len(self.farm_ids))/count).astype(cp.float32)}

    def receipt(self):
        status=cp.asnumpy(self.status)
        if np.any(status):raise ValueError(f'MAV native-undefined input: graph cycle (-2) or rotor outside vertical grid (turbine): {status.tolist()}')
        return dict(schema='gpuwm-wind-receipt.v1',scheme='mav',turbine_count=self.nt,
                    table_sha256=self.table_sha256,execution='resident graph frontiers and parallel rotors',
                    occupied_columns=int(self.active_columns.size),cost=self.cost.receipt(),options=self.options,
                    turbine_ids=self.ids,farm_ids=self.farm_ids,input_provenance=self.data.get('provenance',{}))

    def write_replay(self,path):
        return None  # Native-captured qualification is supplied by tools/wind_mav_wrf471_oracle.
