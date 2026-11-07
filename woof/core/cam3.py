"""WRF 4.6.1 CAM radiation on the common column and interval interfaces."""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import json,os
import numpy as np
from woof.core.cam3_tables import cached_tables
from woof.core.cam3_memory import DEFAULT_WORKERS, worker_count, column_parameters

def absorption_due(elapsed,dt,period,frequency,*,first,adaptive=False,ra_call_offset=0):
    """WRF radiation_driver STEPABS and adaptive boundary rules."""
    if first or frequency==0:return True
    if adaptive:
        return elapsed+dt >= (int(elapsed/frequency)+1)*frequency
    interval=max(1,int(np.floor(np.float32(frequency)/np.float32(period)+np.float32(.5))))*period
    steps=max(1,int(np.floor(interval/float(dt)+.5)))
    return steps==1 or (int(np.floor(elapsed/float(dt)+.5)) % steps)==ra_call_offset % steps

class Cam3Radiation:
    # Workers come from cam3_memory.worker_count at DEFAULT_WORKERS, the
    # count preflight prices; callers do not impose the RRTMGP chunk.
    sizes_own_columns=True
    publishes_olr=True
    publishes_cldfra=True
    def __init__(self,start_time,latitude_deg,longitude_deg,*,table_root=None,column_chunk=DEFAULT_WORKERS,icloud=1,longwave=True,shortwave=True):
        import cupy as cp
        self.start_time=start_time;self.latitude_deg=cp.ascontiguousarray(cp.asarray(latitude_deg),dtype=cp.float32);self.longitude_deg=cp.ascontiguousarray(cp.asarray(longitude_deg),dtype=cp.float32)
        if self.latitude_deg.shape!=self.longitude_deg.shape:raise ValueError('CAM radiation geography must share one grid')
        self.longwave=bool(longwave);self.shortwave=bool(shortwave);self.publishes_olr=self.longwave
        self.icloud=int(icloud);self.column_chunk=min(int(column_chunk or DEFAULT_WORKERS),DEFAULT_WORKERS);self.update_count=0;self._abs_time=None;self._retained={};self._state=None;self._input_constants={}
        if self.column_chunk<1:raise ValueError('CAM column chunk must be positive')
        if table_root is None:table_root=os.environ.get('WOOF_CAM_TABLE_ROOT',str(Path.home()/'.woof/tables/cam3'))
        from woof.table_assets import stage_cam3_tables, CAM3_TABLE_ASSETS
        root=Path(table_root).expanduser()
        if any(not (root/asset.filename).is_file() for asset in CAM3_TABLE_ASSETS):
            if stage_cam3_tables(root=root): raise RuntimeError("CAM table acquisition failed before radiation initialization")
        self.tables=cached_tables(str(root.resolve()))
        metadata=json.loads((Path(__file__).parent/'kernels/cam3_emissivity.json').read_text())['camrad'];self._metadata=metadata['parameters']
        from woof.core.kernels import get_kernel
        source=cp.asarray(self.tables.ozone_columns(cp.asnumpy(self.latitude_deg)).reshape(-1,59,12));self._ozone=cp.empty((self.latitude_deg.size,59*13),cp.float32)
        get_kernel('cam3_prepare','cam_monthly_ozone')(((self.latitude_deg.size+63)//64,),(64,),(np.int64(self.latitude_deg.size),source,self._ozone))
    @staticmethod
    def _columns(array):
        import cupy as cp
        return cp.ascontiguousarray(array.transpose(1,2,0).reshape(-1,array.shape[0]),dtype=cp.float32)
    def __call__(self,*,atmosphere,fields,state,cfg):
        import cupy as cp
        from woof.core.kernels import get_kernel
        from woof.core.physics import RadiationResult,_physics_period_seconds
        from woof.core.swint import calendar_scalars
        from woof.core.rrtmgp import cal_cldfra1, cal_cldfra2
        from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
        t=atmosphere['temperature'];nz,ny,nx=t.shape;ncol=ny*nx
        if self.latitude_deg.shape!=(ny,nx):raise ValueError('CAM geography must match the model grid')
        julian,xtime,gmt=calendar_scalars(self.start_time,state.elapsed_seconds);minutes=np.float32(cfg.radt if cfg.radt>0 else cfg.radt_minutes);period=_physics_period_seconds(minutes,cfg)
        valid=self.start_time+timedelta(seconds=float(state.elapsed_seconds));degrad=np.float32(np.float32(3.141592653589793)/np.float32(180.))
        from woof.core.solar import solar_geometry_device
        pair,cosine,_hrang=solar_geometry_device(julian,np.float32(xtime+np.float32(minutes*0.5)),gmt,self.latitude_deg,self.longitude_deg,generation=getattr(cfg,'solar_generation','wrf_480'))
        cosine=cosine.reshape(-1);solar=cp.broadcast_to(pair[1:2],(ncol,)).copy();grid=((ncol+63)//64,);block=(64,)
        if self._state is None:
            mxaer=cp.empty(1,cp.int32);get_kernel('cam3_prepare','cam_background_levels')((1,),(1,),(np.int32(nz),state.znu,np.float32(np.float32(state.p_top)/np.float32(1000.)),mxaer))
            # module_model_constants.F: EP_2 is default-REAL R_D/R_V.
            eps=np.float32(np.float32(287.)/np.float32(461.6))
            physical=np.array([9.81,1004.5,float(eps),5.67051e-8,287.,461.6,float(eps),2.501e6,3.336e5,float(np.float32(461.6)),1004.5,float(np.float32(273.16))],np.float64)
            from woof.numerics import is_strict
            self._state=cp.asarray(self.tables.initialized_state(physical,int(cp.asnumpy(mxaer)[0]),fast=not is_strict()))
        zero=cp.zeros_like(t);species={name:getattr(state,name,None) for name in ('qc','qr','qi','qs','qg')};species={name:(zero if value is None else value) for name,value in species.items()}
        f_qi,f_qs=legacy_cloud_fraction_flags(int(cfg.mp_physics))
        if self.icloud == 2:
            cloud=cal_cldfra2(self._columns(species['qc']),self._columns(species['qi']),f_qc=int(cfg.mp_physics)!=0,f_qi=f_qi)
        else:
            cloud=cal_cldfra1(self._columns(atmosphere['qv']),self._columns(species['qc']),self._columns(species['qi']),self._columns(species['qs']),self._columns(t),self._columns(atmosphere['pressure']),f_qc=True,f_qi=f_qi,f_qs=f_qs,mp_physics=int(cfg.mp_physics),f_ice_phy=self._columns(state.f_ice_phy) if int(cfg.mp_physics)==95 else None)
        from woof.core.mynn_radiation import merge_mynn_bl_clouds, mynn_bl_cloud_active, wrf_itimestep
        if self.icloud == 1 and mynn_bl_cloud_active(cfg.bl_pbl_physics,cfg.icloud_bl):
            liquid=self._columns(species['qc']).copy();ice=self._columns(species['qi']).copy()
            liquid,ice,cloud=merge_mynn_bl_clouds(liquid,ice,cloud,qc_bl=self._columns(fields['qc_bl']),qi_bl=self._columns(fields['qi_bl']),cldfra_bl=self._columns(fields['cldfra_bl']),bl_pbl_physics=cfg.bl_pbl_physics,icloud_bl=cfg.icloud_bl,itimestep=wrf_itimestep(state.elapsed_seconds,cfg.dt),bl_mynn_version=getattr(cfg,'bl_mynn_version','wrf_461'),temperature=self._columns(t))
            species['qc']=liquid.reshape(ny,nx,nz).transpose(2,0,1);species['qi']=ice.reshape(ny,nx,nz).transpose(2,0,1)
        if self.icloud==0:cloud.fill(0);species={name:zero for name in species}
        # Native module_radiation_driver merges Deng after resolved clouds,
        # including the clear resolved-cloud setting (WRF 4.7.1:1566-1577).
        if int(getattr(cfg,'shcu_physics',0)) == 5:
            from woof.core.column_cumulus import merge_column_cumulus_clouds
            liquid,ice,cloud=merge_column_cumulus_clouds(
                self._columns(species['qc']),self._columns(species['qi']),cloud,
                fields=fields,cfg=cfg,columns=self._columns)
            species['qc']=liquid.reshape(ny,nx,nz).transpose(2,0,1)
            species['qi']=ice.reshape(ny,nx,nz).transpose(2,0,1)
        inputs={'t_phy':self._columns(t),'p_phy':self._columns(atmosphere['pressure']),'p8w':self._columns(atmosphere['p_interface']),'pi_phy':self._columns(atmosphere['exner']),'rho_phy':self._columns(atmosphere['rho']),'dz8w':self._columns(atmosphere['dz']),'z':self._columns(np.float32(0.5)*(atmosphere['z_interface'][:-1]+atmosphere['z_interface'][1:])),'qv3d':self._columns(atmosphere['qv']),'cldfra':cloud}
        inputs.update({name+'3d':self._columns(value) for name,value in species.items()})
        for name,key in [('tsk','tsk'),('emiss','emiss'),('albedo','albedo'),('xland','xland'),('xice','xice'),('snow','snow')]:
            inputs[name]=cp.ascontiguousarray(fields[key],dtype=cp.float32).reshape(ncol,1)
        if not self._input_constants:
            previous=cp.empty((ncol,29*13),cp.float32);following=cp.empty_like(previous);pressure=cp.empty((ncol,1),cp.float32)
            get_kernel('cam3_prepare','cam_aerosol_inputs')(grid,block,(np.int64(ncol),np.int32(29),cp.asarray(self.tables.hybi),previous,following,pressure))
            self._input_constants={'pin0':cp.broadcast_to(cp.asarray(self.tables.pin),(ncol,59)).copy(),'m_hybi0':cp.broadcast_to(cp.asarray(self.tables.hybi),(ncol,29)).copy(),'aerosolcp':previous,'aerosolcn':following,'m_psp':pressure,'m_psn':pressure}
        inputs.update(self._input_constants)
        inputs.update({'xlat':self.latitude_deg.reshape(ncol,1),'xlong':self.longitude_deg.reshape(ncol,1),'coszen':cosine[:,None],'solcon':solar[:,None],'ozmixm':self._ozone})
        for name in ('alswvisdir','alswvisdif','alswnirdir','alswnirdif'):inputs[name]=inputs['albedo']
        freq=float(getattr(cfg,'cam_abs_freq_s',21600.));recompute=absorption_due(float(state.elapsed_seconds),float(cfg.dt),period,freq,first=self._abs_time is None,adaptive=bool(getattr(cfg,'use_adaptive_time_step',False)),ra_call_offset=cfg.ra_call_offset)
        params={'ids':1,'ide':2,'jds':1,'jde':2,'kds':1,'kde':nz+1,'ims':1,'ime':1,'jms':1,'jme':1,'kms':1,'kme':nz+1,'its':1,'ite':1,'jts':1,'jte':1,'kts':1,'kte':nz,'levsiz':59,'num_months':13,'paerlev':29,'naer_c':13,'cam_abs_dim1':4,'cam_abs_dim2':nz+1,'n_cldadv':3,'dolw':int(self.longwave),'dosw':int(self.shortwave),'doabsems':int(recompute),'ghg_input':int(getattr(cfg,'ghg_input',1)),'sf_surface_physics':int(cfg.sf_surface_physics),'f_qv':1,'f_qc':1,'f_qr':int(getattr(state,'qr',None) is not None),'f_qi':int(f_qi),'f_qs':int(f_qs),'f_qg':int(getattr(state,'qg',None) is not None),'gmt':gmt,'julday':valid.timetuple().tm_yday,'julian':julian,'yr':valid.year,'dt':np.float32(cfg.dt),'xtime':xtime,'radt':minutes,'degrad':degrad}
        gas=cp.asarray(self.tables.gases(valid.year,julian));inputs['gases']=cp.broadcast_to(gas,(ncol,5)).copy()
        # One parameter launch and one zeroed output arena replace a separate
        # allocation/zero/fill for each of the 121 ABI arguments.
        parameters=column_parameters(nz);parameter_types={name:dtype for name,dtype,_ in parameters}
        param_names=list(params)
        param_bits=np.array([np.asarray(params[name],dtype=np.int32 if parameter_types[name]=='int' else np.float32).view(np.uint32).item() for name in param_names],np.uint32)
        param_source=cp.asarray(param_bits);param_values=cp.empty((len(params),ncol),cp.uint32)
        get_kernel('cam3_prepare','cam_fill_parameters')(((param_values.size+255)//256,),(256,),(np.int64(param_values.size),np.int32(ncol),param_source,param_values))
        param_views={name:param_values[i].reshape(ncol,1) for i,name in enumerate(param_names)}
        widths={'int':4,'float':4,'double':8};arena_bytes=0;arena_layout={}
        for name,_,count in parameters:
            if name in params or name in self._retained:continue
            if name in inputs and inputs[name].shape==(ncol,count) and name!='cldfra':continue
            arena_bytes=(arena_bytes+7)&~7;arena_layout[name]=(arena_bytes,count);arena_bytes+=ncol*count*widths[parameter_types[name]]
        arena=cp.zeros(arena_bytes,cp.uint8)
        outputs={};buffers=[]
        for name,kind,count in parameters:
            dtype={'int':cp.int32,'float':cp.float32,'double':cp.float64}[kind]
            if name in params:value=param_views[name].view(dtype)
            elif name in self._retained:value=self._columns(self._retained[name])
            elif name not in arena_layout:value=inputs[name]
            else:
                offset,count=arena_layout[name];value=cp.ndarray((ncol,count),dtype=dtype,memptr=arena.data+offset)
                if name in inputs:
                    supplied=inputs[name];value[:,:supplied.shape[1]]=supplied
            buffers.append(value);outputs[name]=value
        workers=worker_count(ncol,self.column_chunk)
        from woof.core.cam3_launch import launch_columns
        launch_columns(nz,self._state,buffers,workers,refresh=recompute and self.longwave,shortwave=self.shortwave,longwave=self.longwave)
        for name in ('abstot_3d','absnxt_3d','emstot_3d'):
            grid=outputs[name].reshape(ny,nx,-1).transpose(2,0,1)
            if name not in self._retained:self._retained[name]=cp.ascontiguousarray(grid)
            else:self._retained[name][...]=grid
        if recompute or not self.longwave:self._abs_time=float(state.elapsed_seconds)
        self.update_count+=1
        self._sync_stream_clock()
        layer=lambda name:cp.ascontiguousarray(outputs[name][:,:nz].reshape(ny,nx,nz).transpose(2,0,1))
        surface=lambda name:outputs[name].reshape(ny,nx)
        from woof.core.cam3_memory import CAM_FIELDS_2D, CAM_FIELDS_3D
        from woof.core.physics_inventory import RADIATION_BOUNDARY_FLUXES
        for name in CAM_FIELDS_2D:
            if (self.longwave if name=='lwcf' else self.shortwave): fields[name][...] = surface(name)
        for name in CAM_FIELDS_3D:
            if (self.longwave if name in ('cemiss','rthratenlwc') else self.shortwave): fields[name][...] = layer(name)
        return RadiationResult(boundary_fluxes={name:surface(name) for name in RADIATION_BOUNDARY_FLUXES if (self.longwave if name.startswith('lw') else self.shortwave)},rthratenlw=layer('rthratenlw'),rthratensw=layer('rthratensw'),swdown=surface('swdnb'),glw=surface('glw'),gsw=surface('gsw'),coszen=cosine.reshape(ny,nx),olr=surface('olr') if self.longwave else None,cldfra=cp.ascontiguousarray(cloud.reshape(ny,nx,nz).transpose(2,0,1)),swddir=surface('swddir'),swddif=surface('swddif'))
    def wrf_warm_start_rebuild(self):
        """Native WRF does not restart-write CAM's three absorption arrays.

        Registry.EM_COMMON:2068-2070 omits the restart flag. camrad:522-524
        detects the zero cache and forces an absorption refresh on its next
        longwave call. Held heating is restored separately by the driver.
        woof checkpoints instead preserve the cache and its exact clock.
        """
        self._retained.clear();self._abs_time=None;self.update_count=0;self._state=None
        self._sync_stream_clock()
        return {'absorption_clock':
                'native WRF omits CAM absorption arrays from wrfrst; '
                'camrad:522-524 forces refresh at the next longwave call'}
    checkpoint_setup_arrays=frozenset({'latitude_deg','longitude_deg','_ozone','_state','_stream_clock'})
    checkpoint_setup_containers=frozenset({'tables','_input_constants'})
    checkpoint_serialized_containers=frozenset({'_retained'})
    _retained_names=frozenset({'abstot_3d','absnxt_3d','emstot_3d'})
    def checkpoint_manifest(self):
        if set(self._retained) not in (set(),set(self._retained_names)):
            raise ValueError('CAM absorption checkpoint must carry all three retained arrays')
        # Keep the checkpoint's original column ABI; streaming uses stable
        # C-contiguous horizontal planes instead.
        result={name:value.transpose(1,2,0).reshape(-1,value.shape[0]).copy()
                for name,value in self._retained.items()} if self.update_count else {}
        result['absorption_clock']=np.array([-1. if self._abs_time is None else self._abs_time,float(self.update_count)],np.float64)
        return result
    def streaming_manifest(self, *, state_shape=None):
        import cupy as cp
        if not self._retained and state_shape is not None:
            nz,ny,nx=state_shape
            for field in self._metadata:
                if field['name'] in self._retained_names:
                    size=eval(field.get('count_expression',str(field['count'])),
                              {'__builtins__':{}},{'CAM_LEVELS':nz})
                    self._retained[field['name']]=cp.zeros((size,ny,nx),cp.float32)
        result=dict(self._retained)
        # The scalar hook drives cadence without a device-to-host read.
        # Two constant planes also retain the original clock member in a
        # streamed checkpoint, so a resident restart can read that archive.
        if getattr(self,'_stream_clock',None) is None:
            self._stream_clock=cp.empty((2,*self.latitude_deg.shape),cp.float64)
            self._sync_stream_clock()
        result['absorption_clock']=self._stream_clock
        return result
    def _sync_stream_clock(self):
        clock=getattr(self,'_stream_clock',None)
        if clock is not None:
            clock[0].fill(-1. if self._abs_time is None else self._abs_time)
            clock[1].fill(self.update_count)
    def streaming_scalars(self):
        return {'abs_time':self._abs_time,'update_count':self.update_count}
    def streaming_restore_scalars(self,values):
        self._abs_time=values['abs_time'];self.update_count=int(values['update_count'])
        self._sync_stream_clock()
    def checkpoint_validate(self,payload,state):
        nz,ny,nx=state.p.shape
        clock=np.asarray(payload.get('absorption_clock'))
        if clock.ndim==3:
            if clock.shape!=(2,ny,nx) or not np.all(clock==clock[:,:1,:1]):
                raise ValueError('CAM streamed absorption clock must be uniform')
            clock=clock[:,0,0]
        if clock.shape!=(2,) or clock.dtype!=np.float64 or not np.isfinite(clock).all():
            raise ValueError('CAM absorption clock must contain two finite binary64 values')
        time,count=clock
        if count<0 or count!=int(count) or (time<0 and time!=-1.):raise ValueError('invalid CAM absorption clock')
        expected={'absorption_clock'} | (set(self._retained_names) if count or self._retained_names.intersection(payload) else set())
        if set(payload)!=expected or (count and time<0):raise ValueError('CAM checkpoint omits or adds retained absorption state')
        for field in self._metadata:
            name=field['name']
            if name not in self._retained_names or name not in payload:continue
            size=eval(field.get('count_expression',str(field['count'])),{'__builtins__':{}},{'CAM_LEVELS':nz})
            array=payload[name]
            if array.shape not in ((ny*nx,size),(size,ny,nx)) or array.dtype!=np.float32:raise ValueError('CAM retained absorption shape or dtype differs: '+name)
    def checkpoint_restore(self,payload,state,asarray):
        self.checkpoint_validate(payload,state)
        clock=np.asarray(payload['absorption_clock'])
        time,count=clock if clock.ndim==1 else clock[:,0,0]
        self._abs_time=None if time==-1. else float(time);self.update_count=int(count)
        _,ny,nx=state.p.shape
        self._retained={name:(asarray(payload[name]).copy() if payload[name].ndim==3
                             else asarray(payload[name]).reshape(ny,nx,-1).transpose(2,0,1).copy())
                        for name in self._retained_names if name in payload}
        self._state=None
        self._sync_stream_clock()
    @property
    def restart_identity(self):
        from woof.table_assets import CAM3_TABLE_ASSETS
        return {'algorithm':'wrf-v4.6.1-cam3-v1','table_assets':{a.filename:a.sha256 for a in CAM3_TABLE_ASSETS},'icloud':self.icloud,'spectra':{'lw':self.longwave,'sw':self.shortwave},'above_atmosphere_policy':'CAM model column'}
