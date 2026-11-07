"""Resident CUDA port of WRF V3.5's windspec analytic turbine drag."""
import hashlib
import json
import cupy as cp
import numpy as np
from woof.core.kernels import get_kernel
from woof.wind_legacy import read_spec, DEFAULTS


class LegacyFarm:
    scheme='fitch_legacy'

    def __init__(self,cfg,shape,*,static=None,phb=None,**kwargs):
        self.shape=tuple(shape);nz,ny,nx=self.shape;self.nc=nx*ny
        self.dx=np.float32(cfg.dx);self.dt=np.float32(cfg.dt)
        self.advect=np.int32(cfg.bl_mynn_tkeadvect)
        if cfg.windturbines_spec=='ideal':
            # The rectangular row expansion is a Rust data path as well.
            import subprocess
            from woof.wind_inputs import table_binary
            request={k:getattr(cfg,k) for k in DEFAULTS}
            request.update(nx=nx,ny=ny,grid_id=cfg.grid_id)
            result=subprocess.run([str(table_binary()),'--wind-ideal',json.dumps(request)],
                                  capture_output=True,check=True,timeout=90)
            data=json.loads(result.stdout)
            self.table_sha256=hashlib.sha256(result.stdout).hexdigest()
            rows=data['turbines'];col=cp.asarray([r['grid_y']*nx+r['grid_x'] for r in rows],cp.int32)
            self.density=np.float32(cfg.td_turbpercell)
        else:
            data,self.table_sha256=read_spec(cfg.windturbines_spec)
            rows=[r for r in data['turbines'] if r['grid_id']==cfg.grid_id]
            self.density=np.float32(1.)
            if rows:
                if static is None or any(static.get(k) is None for k in ('XLAT_V','XLONG_U','XLAT_U')):
                    raise ValueError('windturbines_spec geographic files require native XLAT_U, XLAT_V and XLONG_U staggered coordinates')
                latu=cp.asarray(static['XLAT_U'],cp.float32);lonu=cp.asarray(static['XLONG_U'],cp.float32)
                if not bool(cp.any(latu!=0) | cp.any(lonu!=0)):
                    raise ValueError('WRF V3.5 pseudo-real windspec leaves turbine i/j=-999; native cannot activate this documented mode; use windturbines_spec=ideal')
                latv=cp.ascontiguousarray(cp.asarray(static['XLAT_V'],cp.float32))
                lonu=cp.ascontiguousarray(lonu,cp.float32)
                if latv.shape!=(ny+1,nx) or lonu.shape!=(ny,nx+1):
                    raise ValueError('windspec staggered coordinates do not match the native domain')
                ll=cp.asarray([[r['latitude'],r['longitude']] for r in rows],cp.float32)
                col=cp.empty(len(rows),cp.int32)
                get_kernel('wind_legacy','legacy_locate')(((len(rows)+127)//128,),(128,),
                    (ll,latv,lonu,col,np.int32(len(rows)),np.int32(nx),np.int32(ny)))
                positions=cp.asnumpy(col) # preparation only
                rows=[r for r,c in zip(rows,positions) if c>=0]
                col=cp.asarray(positions[positions>=0],cp.int32)
            else:col=cp.empty(0,cp.int32)
        self.data=data;self.nt=len(rows);self.ids=[r['id'] for r in rows]
        self.col=cp.ascontiguousarray(col)
        names=('hub_height_m','rotor_diameter_m','standing_ct','rated_power_mw','cut_in_m_s','cut_out_m_s')
        self.params=cp.asarray([[r[k] for k in names] for r in rows],cp.float32)
        self.members=cp.ascontiguousarray(cp.argsort(col,kind='stable'),cp.int32)
        counts=cp.bincount(col,minlength=self.nc)
        self.ptr=cp.concatenate((cp.zeros(1,cp.int32),cp.cumsum(counts,dtype=cp.int32)))
        self.active_columns=cp.ascontiguousarray(cp.flatnonzero(counts),cp.int32)
        self.grid=(int(self.active_columns.size),);self.block=(max(32,1<<(nz-1).bit_length()),)
        if self.block[0]>1024:raise ValueError('legacy turbine CUDA supports at most 1024 mass levels')
        if phb is None:raise ValueError('legacy turbine physics requires the native base geopotential PHB')
        self.phb=cp.ascontiguousarray(cp.broadcast_to(cp.asarray(phb,cp.float32), (nz+1,ny,nx)))
        self.tke_rate=cp.zeros(shape,cp.float32);self.tke_increment=cp.zeros_like(self.tke_rate)
        self.kernel=get_kernel('wind_legacy','legacy_drag')
        self.prefix=(self.active_columns,self.ptr,self.members,self.params)
        self.calls=0

    def apply_mynn(self,atmosphere,fields,rates,*,step_seconds=None):
        self.calls+=1
        if step_seconds is not None:self.dt=np.float32(step_seconds)
        if self.nt:
            self.kernel(self.grid,self.block,self.prefix+(atmosphere['u'],atmosphere['v'],atmosphere['dz'],self.phb,
                rates['du'],rates['dv'],fields['qke'],fields.get('qke_adv',fields['qke']),self.tke_rate,self.tke_increment,
                np.int32(self.nc),np.int32(self.shape[0]),self.dx,self.dt,self.density,self.advect))
        # Native turbine_drag copies the complete carrier, including columns
        # without turbines. This device copy never waits on the host.
        if self.advect and 'qke_adv' in fields:
            cp.copyto(fields['qke_adv'],fields['qke'])

    def history_fields(self):
        return {'TKE_WINDFARM':self.tke_increment,'WIND_TKE_SOURCE':self.tke_rate,
                'WIND_TKE_SOURCE_MAX':cp.max(self.tke_rate,axis=0)}

    def receipt(self):
        return dict(schema='gpuwm-wind-receipt.v1',scheme=self.scheme,turbine_count=self.nt,
            turbine_ids=self.ids,table_sha256=self.table_sha256,execution='fused_sparse_columns',
            occupied_columns=int(self.active_columns.size),calls=self.calls,
            input_provenance=self.data.get('provenance',{}),timing_source='separate CUDA graph stage profile')

    def write_replay(self,path):return None

    def sample(self,*args):return None
