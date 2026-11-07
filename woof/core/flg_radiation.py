"""WRF RAD_FLG packing, full rad_all column, and radiation carriers on CUDA.

The stock WRF entry fixes its offline controls: old gas quadrature, mixed
ice optics, three maximum-random cloud groups, prescribed ozone, no aerosol,
and the mean effective ice size. Tables are resident for the adapter's life.
"""
from __future__ import annotations
import numpy as np


class FLGRadiation:
    publishes_olr = True
    publishes_cldfra = True
    glw_provenance = 'scheme'
    diagnostics = ('SWDDNI','SWDDIR','SWDDIF','TLWDN','TLWUP','SLWDN','SLWUP',
                   'TSWDN','TSWUP','SSWDN','SSWUP')

    def __init__(self, start_time, latitude_deg, longitude_deg, *, column_chunk=None):
        import cupy as cp
        from woof.core.flg import load_tables
        from woof.core.kernels import load_module
        self.start_time = start_time
        self.latitude_deg = cp.ascontiguousarray(cp.asarray(latitude_deg,dtype=cp.float32)).ravel()
        self.longitude_deg = cp.ascontiguousarray(cp.asarray(longitude_deg,dtype=cp.float32)).ravel()
        if self.latitude_deg.shape != self.longitude_deg.shape:
            raise ValueError('FLG latitude and longitude shapes differ')
        table = load_tables()
        self.ozone_p = cp.asarray(table['module_ozone.pres'].ravel(order='F'))
        self.ozone_v = cp.asarray(table['module_ozone.ozone'].ravel(order='F'))
        module = load_module('flg_driver')
        self.solar_kernel = module.get_function('flg_solar_driver')
        self.prepare_kernel = module.get_function('flg_prepare_driver')
        self.publish_kernel = module.get_function('flg_publish_driver')
        self.radconst_kernel = load_module('rrtmg_legacy_solar').get_function('rls_radconst')
        self.column = None
        self.update_count = 0

    def restart_identity(self):
        from woof.core.flg import SOURCE_SHA256, TABLE_SHA256
        return {'algorithm':'wrf461-flg-rad-all-v1','source_sha256':SOURCE_SHA256,
                'above_atmosphere_policy':'model-column-top-interface-no-added-cap',
                'table_sha256':TABLE_SHA256,'nice':2,'ngas':0,'npde':2,
                'nfract':0,'ninho':0,'nozone':1,'naero':0,
                'solar':'RAD_FLG interval midpoint without equation of time'}

    def _allocate(self,ncol,nz):
        import cupy as cp
        from woof.core.flg import FLGColumn
        self.column = FLGColumn(ncol,nz)
        self.inputs = {name:cp.empty((ncol,nz+1),cp.float32) for name in FLGColumn.profile_names}
        self.inputs.update({name:cp.empty((ncol,nz),cp.float32) for name in FLGColumn.layer_names})
        self.inputs.update({name:cp.empty((ncol,width),cp.float32) for name,width in (('albedo',6),('emissivity',12))})
        self.inputs.update({name:cp.empty(ncol,cp.float32) for name in FLGColumn.scalar_names})
        self.heat_lw = cp.empty((ncol,nz),cp.float32)
        self.heat_sw = cp.empty_like(self.heat_lw)
        self.heat_total = cp.empty_like(self.heat_lw)
        self.surface = cp.empty((15,ncol),cp.float32)
        self.zeros = cp.zeros((ncol,nz),cp.float32)
        self.solar = cp.empty(2,cp.float32)

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from woof.core import constants as c
        from woof.core.physics import RadiationResult
        from woof.core.rrtm_lw import _t8w_columns
        from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags, _DEGRAD, _DPD
        from woof.core.flg_cloud_fraction import cal_cldfra1
        from woof.core.swint import calendar_scalars
        from woof.core.mynn_radiation import merge_mynn_bl_clouds, mynn_bl_cloud_active, wrf_itimestep
        nz,ny,nx = atmosphere['pressure'].shape
        ncol = ny*nx
        if self.latitude_deg.size != ncol:
            raise ValueError('FLG geography differs from the model column count')
        if self.column is None:self._allocate(ncol,nz)
        if (self.column.columns,self.column.layers)!=(ncol,nz):
            raise ValueError('FLG adapter cannot change its column shape between updates')
        def columns(value):
            return cp.ascontiguousarray(cp.asarray(value,dtype=cp.float32).reshape(-1,ncol).T)
        moist = {name:columns(getattr(state,name)) if getattr(state,name,None) is not None else self.zeros
                 for name in ('qv','qc','qr','qi','qs','qg')}
        temp,pres = columns(atmosphere['temperature']),columns(atmosphere['pressure'])
        fqi,fqs = ((True,True) if int(cfg.mp_physics)==7 else legacy_cloud_fraction_flags(int(cfg.mp_physics)))
        cf = (cal_cldfra1(moist['qv'],moist['qc'],moist['qi'],moist['qs'],temp,pres,
                         f_qc=True,f_qi=fqi,f_qs=fqs,
                         f_ice_phy=columns(state.f_ice_phy) if int(cfg.mp_physics)==95 else None) if cfg.icloud else self.zeros)
        active = mynn_bl_cloud_active(getattr(cfg,'bl_pbl_physics',0),getattr(cfg,'icloud_bl',0))
        if active:
            moist['qc']=moist['qc'].copy();moist['qi']=moist['qi'].copy()
        moist['qc'],moist['qi'],cf = merge_mynn_bl_clouds(moist['qc'],moist['qi'],cf,
            qc_bl=columns(fields['qc_bl']) if active else None,
            qi_bl=columns(fields['qi_bl']) if active else None,
            cldfra_bl=columns(fields['cldfra_bl']) if active else None,
            bl_pbl_physics=getattr(cfg,'bl_pbl_physics',0),icloud_bl=getattr(cfg,'icloud_bl',0),
            itimestep=wrf_itimestep(state.elapsed_seconds,cfg.dt) if active else 1,
            bl_mynn_version=getattr(cfg,'bl_mynn_version','wrf_461'),temperature=temp if active else None)
        if cfg.shcu_physics == 5:
            from woof.core.column_cumulus import merge_column_cumulus_clouds
            moist['qc'],moist['qi'],cf=merge_column_cumulus_clouds(
                moist['qc'],moist['qi'],cf,fields=fields,cfg=cfg,columns=columns)
        tw = _t8w_columns(cp,temp,columns(atmosphere['z_interface']),
                          cp.asarray(state.fnm,dtype=cp.float32),cp.asarray(state.fnp,dtype=cp.float32))
        julian,xtime,gmt = calendar_scalars(self.start_time,state.elapsed_seconds)
        self.radconst_kernel((1,),(1,),(np.float32(julian),_DEGRAD,_DPD,self.solar))
        launch = (((ncol+31)//32,),(32,))
        # RAD_FLG has its own solar angle; calc_coszen includes an EOT term
        # absent from this scheme. RADT here is the requested minute value.
        minutes = np.float32(cfg.radt if cfg.radt>0 else cfg.radt_minutes)
        self.solar_kernel(*launch,(np.int32(ncol),np.float32(xtime),minutes,np.float32(gmt),_DEGRAD,
            self.latitude_deg,self.longitude_deg,self.solar,self.inputs['coszen']))
        order = self.column.profile_names+self.column.layer_names+('albedo','emissivity')+self.column.scalar_names
        self.prepare_kernel(*launch,(np.int32(ncol),np.int32(nz),np.int32(int(julian)+1),
            np.int32(fqi),np.int32(int(cfg.mp_physics)==1),np.int32(cfg.icloud!=0),
            columns(atmosphere['p_interface']),cp.ascontiguousarray(tw),temp,
            *[moist[name] for name in ('qv','qc','qr','qi','qs','qg')],columns(atmosphere['rho']),
            self.latitude_deg,*[columns(fields[name]).ravel() for name in ('albedo','emiss','tsk')],
            self.inputs['coszen'],self.solar,self.ozone_p,self.ozone_v,*[self.inputs[name] for name in order]))
        output = self.column(**self.inputs)
        factor = np.float32(np.float32(np.float32(.01)*np.float32(c.G))/np.float32(c.CP))
        self.publish_kernel(*launch,(np.int32(ncol),np.int32(nz),factor,columns(atmosphere['exner']),
            *[output[name] for name in ('dts','dtir','dt','fds','fus','fdir','fuir','direct','diffuse')],
            self.inputs['coszen'],self.heat_sw,self.heat_lw,self.heat_total,self.surface,self.column.errors))
        def mass(value):return cp.ascontiguousarray(value.T).reshape(nz,ny,nx)
        diagnostic = {name:self.surface[index].reshape(ny,nx) for name,index in
                      zip(self.diagnostics,(4,5,6,7,8,9,10,11,12,13,14))}
        self.update_count += 1
        from woof.core.wrf_radiation_clock import wrf_radiation_coszen
        driver_coszen = wrf_radiation_coszen(self.start_time,state.elapsed_seconds,minutes,
                                            self.latitude_deg,self.longitude_deg).reshape(ny,nx)
        return RadiationResult(rthratenlw=mass(self.heat_lw),rthratensw=mass(self.heat_sw),
            swdown=self.surface[0].reshape(ny,nx),glw=self.surface[1].reshape(ny,nx),
            gsw=self.surface[2].reshape(ny,nx),olr=self.surface[3].reshape(ny,nx),
            coszen=driver_coszen,swddir=diagnostic['SWDDIR'],
            swddif=diagnostic['SWDDIF'],cldfra=mass(cf),flux_diagnostics=diagnostic)


def flg_radiation_vram_bytes(ncol, nz):
    """Resident and peak packing storage; no device access during admission."""
    from woof.core.flg import load_tables
    from pathlib import Path
    import re
    header = (Path(__file__).parent / 'kernels' / 'flg_workspace_generated.cuh').read_text()
    scale = int(re.search(r'#define FLG_WORKSPACE_SCALE (\d+)', header)[1])
    bias = int(re.search(r'#define FLG_WORKSPACE_BIAS (-?\d+)', header)[1])
    tables = load_tables()
    fixed = sum(v.size * 4 for v in tables.values()) + len(tables)*4 + (75+375)*4 + 8
    # Workspace, packed inputs, gas/control vectors, column outputs, adapter
    # carriers, geography and simultaneous mass/column packing temporaries.
    words = scale*(nz+1)+bias + (12*nz+25) + 40 + (11*nz+8) + 1
    words += 4*nz+15+2 + 36*(nz+1)
    return int(fixed + ((int(ncol)+31)//32)*32*words*4)
