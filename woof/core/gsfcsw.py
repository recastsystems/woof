"""WRF 4.7.1 old GSFC shortwave, option 2, on the shared CUDA loader.

The default kernel specializes one-column dimensions and flux storage. Strict
replay retains the ordered native translation. Pack and unpack are device
operations; only small coefficient assets live on the host.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from woof.core.thompson_contract import TableAsset, read_validated_table_asset

TABLE_ROOT = Path(__file__).resolve().parents[1] / 'data/wrf_radiation/gsfcsw'
TABLE_ASSET = TableAsset('coefficients.f32',9896,'a23e21798425039a82cf799f81c638a9f6bb7b0562c586829f05bdff10abc533')
MAX_LAYERS = 128
WORKSPACE_WORDS = 32677
COLUMN_CHUNK = 4096

def workspace_columns(columns):
    return min((int(columns)+31)//32*32,COLUMN_CHUNK)

def workspace_bytes(columns):
    """Bounded call workspace, also charged by forecast preflight."""
    return workspace_columns(columns)*WORKSPACE_WORDS*4

def load_tables(root=None):
    """Refuse changed coefficient words before a physics call."""
    root=TABLE_ROOT if root is None else Path(root)
    try:
        data=read_validated_table_asset(root,TABLE_ASSET)
    except (ValueError,FileNotFoundError) as error:
        raise type(error)('GSFC shortwave coefficient integrity failed: '+str(error)) from error
    return np.frombuffer(data,dtype='<f4')

def packed_columns(inputs,nlay,*,warm_rain=False,f_qi=True,f_qndrop=False,
                   julday,solcon,center_lat,tables=None,initial_heating=None,workspace=None):
    """Evaluate packed bottom-to-top columns and return all five WRF fields.

    Input blocks are T,P,P8W,EXNER,QV,QC,QI,CLDFRA,QNDROP, each nlay+1,
    then COSZEN and ALBEDO. Output blocks are RTHRATEN,TAUCLDI,TAUCLDC,
    each nlay+1, then GSW and RSWTOA. The final layer slots are unused.
    WRF accumulates into RTHRATEN. initial_heating supplies that incoming
    field for a native module replay; spectrum adapters start from zero.
    """
    import cupy as cp
    from woof.core.kernels import load_module
    if isinstance(nlay,bool) or not isinstance(nlay,(int,np.integer)) or not 2<=nlay<=MAX_LAYERS:
        raise ValueError('GSFC shortwave requires 2 to 128 layers; larger columns exceed its allocated CUDA workspace')
    if not 1<=int(julday)<=366: raise ValueError('julday must be 1 to 366')
    if not np.isfinite(solcon) or solcon<=0 or not np.isfinite(center_lat) or abs(center_lat)>90:
        raise ValueError('solcon must be positive and center_lat must be a valid latitude')
    a=cp.ascontiguousarray(inputs,dtype=cp.float32)
    n=nlay+1
    if a.ndim!=2 or a.shape[1]!=9*n+2 or a.shape[0]==0:
        raise ValueError('GSFC packed inputs have shape (ncol,9*(nlay+1)+2)')
    coefficient_words=cp.asarray(load_tables() if tables is None else tables,dtype=cp.float32)
    if coefficient_words.shape!=(TABLE_ASSET.bytes//4,): raise ValueError('GSFC coefficient array has wrong length')
    out=cp.zeros((a.shape[0],3*n+2),dtype=cp.float32)
    if initial_heating is not None:
        incoming=cp.asarray(initial_heating,dtype=cp.float32)
        if incoming.shape!=(a.shape[0],nlay):
            raise ValueError('incoming GSFC heating must have shape (ncol,nlay)')
        out[:,:nlay]=incoming
    kernel=load_module('gsfcsw').get_function('gsfcsw_columns_kernel')
    width=min(a.shape[0],COLUMN_CHUNK)
    if workspace is None:
        workspace=cp.empty((workspace_columns(width),WORKSPACE_WORDS),dtype=cp.float32)
    if workspace.dtype!=cp.float32 or not workspace.flags.c_contiguous or workspace.size<workspace_columns(width)*WORKSPACE_WORDS:
        raise ValueError('GSFC workspace is too small or not contiguous float32')
    for first in range(0,a.shape[0],width):
        count=min(width,a.shape[0]-first)
        kernel(((count+31)//32,),(32,), (coefficient_words,a[first:first+count],out[first:first+count],workspace,np.int32(count),np.int32(nlay),np.int32(warm_rain),np.int32(f_qi),np.int32(f_qndrop),np.int32(julday),np.float32(solcon),np.float32(center_lat)))
    return out

def gsfcsw_columns(temperature,pressure,pressure_interface,exner,qv,qc,qi,
                   cloud_fraction,coszen,albedo,*,qndrop=None,**settings):
    """Device arrays in (ncol,nlay), interfaces in (ncol,nlay+1), bottom up."""
    import cupy as cp
    arrays=[cp.asarray(v,dtype=cp.float32) for v in (temperature,pressure,exner,qv,qc,qi,cloud_fraction)]
    shape=arrays[0].shape
    if len(shape)!=2 or any(a.shape!=shape for a in arrays): raise ValueError('GSFC layer inputs must share (ncol,nlay)')
    nc,nl=shape
    p8=cp.asarray(pressure_interface,dtype=cp.float32)
    mu=cp.asarray(coszen,dtype=cp.float32); alb=cp.asarray(albedo,dtype=cp.float32)
    if p8.shape!=(nc,nl+1) or mu.shape!=(nc,) or alb.shape!=(nc,): raise ValueError('GSFC interface or surface shape differs from columns')
    if qndrop is None: qn=cp.zeros(shape,dtype=cp.float32)
    else:
        qn=cp.asarray(qndrop,dtype=cp.float32)
        if qn.shape!=shape: raise ValueError('QNDROP must share layer shape')
    def pad(a): return cp.concatenate((a,cp.zeros((nc,1),dtype=cp.float32)),axis=1)
    t,p,e,v,c,i,f=arrays
    packed=cp.concatenate([pad(t),pad(p),p8,pad(e),pad(v),pad(c),pad(i),pad(f),pad(qn),mu[:,None],alb[:,None]],axis=1)
    settings.setdefault('f_qndrop',qndrop is not None)
    out=packed_columns(packed,nl,**settings); n=nl+1
    return {'rthraten':out[:,:nl],'taucldi':out[:,n:n+nl],'taucldc':out[:,2*n:2*n+nl],'gsw':out[:,3*n],'rswtoa':out[:,3*n+1]}

@dataclass
class GSFCSWShortwaveRadiation:
    """Stock option-2 spectrum callable; cadence remains in PhysicsDriver."""
    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    icloud: int = 1
    center_lat: float | None = None
    update_count: int = field(default=0,init=False)
    publishes_cldfra = True

    def __post_init__(self):
        import cupy as cp
        if not isinstance(self.start_time,datetime): raise TypeError('radiation_start_time must be a datetime')
        self.latitude_deg=cp.ascontiguousarray(self.latitude_deg,dtype=cp.float32)
        self.longitude_deg=cp.ascontiguousarray(self.longitude_deg,dtype=cp.float32)
        if self.latitude_deg.shape!=self.longitude_deg.shape: raise ValueError('radiation geography shapes differ')
        if self.icloud not in (0,1,2): raise ValueError('icloud must be 0, 1 or 2')
        self.center_lat=(float(cp.asnumpy(cp.mean(self.latitude_deg))) if self.center_lat is None else float(self.center_lat))
        if not np.isfinite(self.center_lat) or abs(self.center_lat)>90:
            raise ValueError("GSFC center latitude must be finite and within 90 degrees")
        self.gsfc_coefficients=cp.asarray(load_tables())
        self.gsfc_workspace=cp.empty((workspace_columns(self.latitude_deg.size),WORKSPACE_WORDS),dtype=cp.float32)

    def __call__(self,*,atmosphere,fields,state,cfg):
        import cupy as cp
        from woof.core.physics import RadiationResult,_physics_period_seconds
        from woof.core.dudhia import wrf_solar_geometry
        from woof.core.rrtmgp import cal_cldfra1, cal_cldfra2
        from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
        t=atmosphere['temperature']; nz,ny,nx=t.shape
        def cols(a): return cp.ascontiguousarray(a.transpose(1,2,0).reshape(ny*nx,a.shape[0]))
        def grid(a): return cp.ascontiguousarray(a.reshape(ny,nx,nz).transpose(2,0,1))
        zero=cp.zeros_like(t)
        qc=getattr(state,'qc',None); qi=getattr(state,'qi',None)
        qc=zero if qc is None else qc; qi=zero if qi is None else qi
        f_qi,f_qs=legacy_cloud_fraction_flags(int(cfg.mp_physics))
        if self.icloud:
            qs=getattr(state,'qs',None)
            if self.icloud == 2:
                cf=grid(cal_cldfra2(cols(qc),cols(qi),f_qc=int(cfg.mp_physics)!=0,f_qi=f_qi))
            else:
                cf=grid(cal_cldfra1(cols(atmosphere['qv']),cols(qc),cols(qi),cols(zero if qs is None else qs),cols(t),cols(atmosphere['pressure']),f_qc=int(cfg.mp_physics)!=0,f_qi=f_qi,f_qs=f_qs,mp_physics=int(cfg.mp_physics),f_ice_phy=cols(state.f_ice_phy) if int(cfg.mp_physics)==95 else None))
        else:
            cf=zero;qc=zero;qi=zero
        from woof.core.mynn_radiation import merge_mynn_bl_clouds,mynn_bl_cloud_active,wrf_itimestep
        if self.icloud == 1 and mynn_bl_cloud_active(cfg.bl_pbl_physics,cfg.icloud_bl):
            qc=qc.copy();qi=qi.copy()
            qc,qi,cf=merge_mynn_bl_clouds(qc,qi,cf,qc_bl=fields['qc_bl'],qi_bl=fields['qi_bl'],cldfra_bl=fields['cldfra_bl'],bl_pbl_physics=cfg.bl_pbl_physics,icloud_bl=cfg.icloud_bl,itimestep=wrf_itimestep(state.elapsed_seconds,cfg.dt),bl_mynn_version=getattr(cfg,'bl_mynn_version','wrf_461'),temperature=t)
        if cfg.shcu_physics == 5:
            from woof.core.column_cumulus import merge_column_cumulus_clouds
            qc,qi,cf=merge_column_cumulus_clouds(qc,qi,cf,fields=fields,cfg=cfg,columns=lambda a:a)
        valid=self.start_time+timedelta(seconds=float(state.elapsed_seconds))
        minutes=cfg.radt if cfg.radt>0 else cfg.radt_minutes
        interval=_physics_period_seconds(minutes,cfg)
        from woof.core.solar import solar_period_seconds
        interval=solar_period_seconds(cfg,interval)
        mu,solcon=wrf_solar_geometry(valid,self.latitude_deg,self.longitude_deg,hour_offset_seconds=0.5*interval,generation=getattr(cfg,'solar_generation','wrf_480'),start_time=self.start_time,elapsed_seconds=state.elapsed_seconds)
        # The atmosphere publishes full interface pressure for each physics call.
        p8=atmosphere['p_interface']
        qn=getattr(state,'nc',None)
        result=gsfcsw_columns(cols(t),cols(atmosphere['pressure']),cols(p8),cols(atmosphere['exner']),cols(atmosphere['qv']),cols(qc),cols(qi),cols(cf),mu.reshape(-1),fields['albedo'].reshape(-1),qndrop=None if qn is None else cols(qn),warm_rain=int(cfg.mp_physics)==1,f_qi=f_qi,solcon=solcon,julday=valid.timetuple().tm_yday,center_lat=self.center_lat,tables=self.gsfc_coefficients,workspace=self.gsfc_workspace)
        for name in ('taucldi','taucldc'):
            value=grid(result[name])
            if name not in fields: fields[name]=cp.zeros_like(t)
            fields[name][...]=value
        if 'rswtoa' not in fields: fields['rswtoa']=cp.zeros((ny,nx),dtype=cp.float32)
        fields['rswtoa'][...]=result['rswtoa'].reshape(ny,nx)
        gsw=result['gsw'].reshape(ny,nx)
        alb=fields['albedo']
        swdown=cp.where(alb<1,gsw/cp.maximum(1-alb,cp.float32(1e-12)),0)
        self.update_count+=1
        return RadiationResult(rthratenlw=zero,rthratensw=grid(result['rthraten']),swdown=swdown,glw=fields['glw'],gsw=gsw,coszen=cp.asarray(mu,dtype=cp.float32),cldfra=cf)

    @property
    def restart_identity(self):
        from woof.io.restart import _array_setup_identity
        coefficients_state=_array_setup_identity(self.gsfc_coefficients)
        return {'algorithm':'wrf-v4.7.1-gsfc-shortwave','coefficients_sha256':TABLE_ASSET.sha256,'coefficients_state':coefficients_state,'center_lat':self.center_lat,'icloud':self.icloud,'supported_path':'no-chem','max_layers':MAX_LAYERS}
