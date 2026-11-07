"""Public WRF solar diagnostic producers and device accumulator ownership."""
from __future__ import annotations
from datetime import timedelta
import numpy as np
from woof.core.native_surface_diagnostics import SOLAR_OUTPUTS, SOLAR_PROFILES, solar
from woof.wrf_exact import ENABLED as WRF_EXACT

_NAMES = 'CLRNIDX SZA GHI_ACCUM CLDFRAC2D WVP LWP IWP SWP WP_SUM LWP_TOT IWP_TOT WP_TOT_SUM RE_QC RE_QI RE_QS RE_QC_TOT RE_QI_TOT TAU_QC TAU_QI TAU_QS TAU_QC_TOT TAU_QI_TOT CBASEHT CTOPHT CBASEHT_TOT CTOPHT_TOT'.split()
# Public Registry/registry.solar_fields metadata, including its fractional
# CLDFRAC2D values labelled percent by native WRF. Preserve both literally.
_NATIVE_META = {'CLDFRAC2D': ('CLDFRAC2D', '%', '2-D MAX CLOUD FRACTION'),
 'WVP': ('WVP', 'kg m-2', 'WATER VAPOR PATH'),
 'LWP': ('LWP', 'kg m-2', 'LIQUID CLOUD WATER PATH'),
 'IWP': ('IWP', 'kg m-2', 'ICE CLOUD WATER PATH'),
 'SWP': ('SWP', 'kg m-2', 'SNOW CLOUD WATER PATH'),
 'WP_SUM': ('WP_SUM', 'kg m-2', 'SUM OF LWP+IWP+SWP'),
 'LWP_TOT': ('LWP_TOT', 'kg m-2', 'LIQUID CLOUD WATER PATH RESOLVED + UNRESOLVED'),
 'IWP_TOT': ('IWP_TOT', 'kg m-2', 'ICE CLOUD WATER PATH RESOLVED + UNRESOLVED'),
 'WP_TOT_SUM': ('WP_TOT_SUM', 'kg m-2', 'SUM OF LWP+IWP+SWP RESOLVED + UNRESOLVED'),
 'RE_QC': ('RE_QC', 'm', 'MASS-WEIGHTED LIQUID CLOUD EFFECTIVE RADIUS'),
 'RE_QI': ('RE_QI', 'm', 'MASS-WEIGHTED ICE EFFECTIVE RADIUS'),
 'RE_QS': ('RE_QS', 'm', 'MASS-WEIGHTED SNOW EFFECTIVE RADIUS'),
 'RE_QC_TOT': ('RE_QC_TOT', 'm', 'MASS-WEIGHTED LIQUID CLOUD EFFECTIVE RADIUS RESOLVED + UNRESOLVED'),
 'RE_QI_TOT': ('RE_QI_TOT', 'm', 'MASS-WEIGHTED ICE EFFECTIVE RADIUS RESOLVED + UNRESOLVED'),
 'TAU_QC': ('TAU_QC', ' ', 'MASS-WEIGHTED LIQUID CLOUD OPTICAL THICKNESS'),
 'TAU_QI': ('TAU_QI', ' ', 'MASS-WEIGHTED ICE OPTICAL THICKNESS'),
 'TAU_QS': ('TAU_QS', ' ', 'MASS-WEIGHTED SNOW OPTICAL THICKNESS'),
 'TAU_QC_TOT': ('TAU_QC_TOT', ' ', 'MASS-WEIGHTED LIQUID CLOUD OPTICAL THICKNESS RESOLVED + UNRESOLVED'),
 'TAU_QI_TOT': ('TAU_QI_TOT', ' ', 'MASS-WEIGHTED ICE OPTICAL THICKNESS RESOLVED + UNRESOLVED'),
 'CBASEHT': ('CBASEHT', 'm agl', 'CLOUD BASE HEIGHT'),
 'CTOPHT': ('CTOPHT', 'm agl', 'CLOUD TOP HEIGHT'),
 'CBASEHT_TOT': ('CBASEHT_TOT', 'm agl', 'CLOUD BASE HEIGHT RESOLVED + UNRESOLVED'),
 'CTOPHT_TOT': ('CTOPHT_TOT', 'm agl', 'CLOUD TOP HEIGHT RESOLVED + UNRESOLVED'),
 'CLRNIDX': ('CLRNIDX', ' ', 'CLEARNESS INDEX'),
 'SZA': ('SZA', 'deg', 'SOLAR ZENITH ANGLE'),
 'GHI_ACCUM': ('GHI_ACCUM', 'J m-2', 'ACCUMULATED GHI')}
HISTORY = {"solar_"+key:_NATIVE_META[name] for key,name in zip(SOLAR_OUTPUTS,_NAMES)}
PERSISTENT=tuple(HISTORY)
WORK=tuple('solar_input_'+k for k in SOLAR_PROFILES if k != 'ph')

def scratch_shapes(cfg):
    if not cfg.solar_diagnostics:return {}
    plane=(cfg.ny,cfg.nx)
    return {**{k:plane for k in HISTORY},**{k:(cfg.nz+1,*plane) for k in WORK},
            'solar_geometry_pair':(2,), 'solar_hrang':plane, 'solar_coszen':plane}

def prepare(state,cfg,atmosphere):
    if not cfg.solar_diagnostics:return
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags, mynn_bl_cloud_active
    s=state.existing_scratch
    zero=s('solar_input_qv')
    n=cfg.nx*cfg.ny
    fqi,fqs=legacy_cloud_fraction_flags(cfg.mp_physics)
    qv=state.qv if state.qv is not None else zero[:cfg.nz]
    qc=state.qc if state.qc is not None else zero[:cfg.nz]
    qi=state.qi if state.qi is not None else zero[:cfg.nz]
    qs=state.qs if state.qs is not None else zero[:cfg.nz]
    # phy_prep precedes radiation, surface and transport, so hold its EOS
    # density and layer depth until the diagnostics driver at solve end.
    cl=s('solar_input_cldfrac3d')[:cfg.nz]
    if WRF_EXACT:
        get_kernel('native_surface_diagnostics','solar_prepare')(((n+127)//128,),(128,),
            (state.alt,qv,state.php,state.phb,s('solar_input_rho'),s('solar_input_dz8w'),
             np.int32(cfg.nz),np.int32(n),np.int32(state.phb.ndim==3)))
        for k,value in (('qc_tot',qc),('qi_tot',qi)):
            s('solar_input_'+k)[:cfg.nz]=value
        get_kernel('native_surface_diagnostics','solar_cloud_fraction')(
            ((cfg.nz*n+127)//128,),(128,),
            (qv,qc,qi,qs,atmosphere['temperature'],state.p,cl,
             np.int32(cfg.nz*n),np.int32(cfg.mp_physics!=0),np.int32(fqi),
             np.int32(fqs),np.int32(cfg.icloud==2),np.int32(cfg.mp_physics in (5,15))))
    else:
        get_kernel('native_surface_diagnostics','solar_prepare_bound')(
            ((cfg.nz*n+127)//128,),(128,),
            (qv,qc,qi,qs,atmosphere['temperature'],state.p,cl,state.alt,state.php,state.phb,
             s('solar_input_rho'),s('solar_input_dz8w'),s('solar_input_qc_tot'),s('solar_input_qi_tot'),
             np.int32(cfg.nz*n),np.int32(n),np.int32(state.phb.ndim==3),np.int32(cfg.mp_physics!=0),
             np.int32(fqi),np.int32(fqs),np.int32(cfg.icloud==2),np.int32(cfg.mp_physics in (5,15))))
    if cfg.icloud==1 and mynn_bl_cloud_active(cfg.bl_pbl_physics,cfg.icloud_bl):
        from woof.core.rrtmg_legacy import _adapter_kernel
        f=state.physics.fields
        _adapter_kernel('rla_mynn')(((cfg.nz*n+127)//128,),(128,),
            (np.int64(cfg.nz*n),s('solar_input_qc_tot'),s('solar_input_qi_tot'),
             f['qc_bl'],f['qi_bl'],f['cldfra_bl'],cl,np.int32(state.elapsed_seconds>float(getattr(state,'domain_start_offset',0.) or 0.)),
             np.uint64(0),np.uint64(0),np.int32(0),np.float32(1.e-6),np.float32(1.e-8),np.float32(.001)))
    # Solar_step runs this clock on every solve, also between radiation
    # calls. Launch its scalar metadata and column kernels without readback.
    from woof.core.kernels import load_module
    from woof.core.rrtmg_legacy import _DEGRAD,_DPD
    now=state.physics.native_diagnostic_epoch+timedelta(seconds=state.elapsed_seconds)
    hour=now.hour+now.minute/60+now.second/3600+now.microsecond/3.6e9
    julian=np.float32(now.timetuple().tm_yday-1+hour/24)
    epoch=state.physics.native_diagnostic_epoch
    gmt=np.float32(epoch.hour+epoch.minute/60+epoch.second/3600+epoch.microsecond/3.6e9)
    xtime=np.float32(np.float32(state.elapsed_seconds/60)+np.float32(cfg.radt if cfg.radt>0 else cfg.radt_minutes)*np.float32(.5))
    mod=load_module('rrtmg_legacy_solar')
    mod.get_function('rls_radconst')((1,),(1,),(julian,_DEGRAD,_DPD,s('solar_geometry_pair')))
    mod.get_function('rls_calc_coszen')(((n+127)//128,),(128,),
        (np.int64(n),state.physics.native_diagnostic_lat,state.physics.native_diagnostic_lon,
         julian,xtime,gmt,_DEGRAD,np.int32(0),s('solar_geometry_pair'),s('solar_coszen'),s('solar_hrang')))
    if 'coszen' in state.physics.fields:state.physics.fields['coszen'][...]=s('solar_coszen')

def update(state,cfg):
    if not cfg.solar_diagnostics:return
    if not WRF_EXACT:return _update_bound(state,cfg)
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags,legacy_scheme_has_req
    s=state.existing_scratch
    profiles={k:s('solar_input_'+k) for k in SOLAR_PROFILES if k not in ('ph','phb')}
    profiles['ph']=state.php
    if state.phb.ndim==3:
        profiles['phb']=state.phb
    else:
        s('solar_input_phb')[...]=state.phb.reshape(-1,1,1)
        profiles['phb']=s('solar_input_phb')
    for k in ('qv','qc','qi','qs'):
        value=getattr(state,k,None)
        if value is not None:profiles[k][:cfg.nz]=value
    radii=legacy_scheme_has_req(cfg.mp_physics,cfg.use_mp_re)
    for k,source,flag in zip(('re_cloud','re_ice','re_snow'),('effc','effi','effs'),radii):
        if flag==1:
            # The stock microphysics wrapper owns the canonical microns.
            from woof.core.rrtmg_legacy import legacy_radius_meters
            profiles[k][:cfg.nz]=legacy_radius_meters(getattr(state,source))
    fqi,fqs=legacy_cloud_fraction_flags(cfg.mp_physics)
    solar(profiles,dict(coszen=s('solar_coszen'),swdnb=state.physics.fields['swdnb'],swdnt=state.physics.fields['swdnt']),
        {k:s('solar_'+k) for k in SOLAR_OUTPUTS},dt=cfg.dt,first=0,last=cfg.nz-1,
        present=(True,cfg.mp_physics!=0,fqi,fqs),radii=radii,_validated=True)

def _update_bound(state,cfg):
    """Read final mass/radius carriers directly; hold only phy_prep inputs."""
    from woof.core.kernels import get_kernel
    from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags,legacy_scheme_has_req
    s=state.existing_scratch;n=cfg.nx*cfg.ny
    zero=s('solar_input_qv')
    def mass(k):
        value=getattr(state,k,None)
        return zero if value is None else value
    radii=legacy_scheme_has_req(cfg.mp_physics,cfg.use_mp_re)
    req=tuple(getattr(state,k) if flag==1 else zero
              for k,flag in zip(('effc','effi','effs'),radii))
    fqi,fqs=legacy_cloud_fraction_flags(cfg.mp_physics)
    get_kernel('native_surface_diagnostics','solar_diag_bound')(((n+127)//128,),(128,),
        (s('solar_input_rho'),s('solar_input_dz8w'),state.php,state.phb,s('solar_input_cldfrac3d'),
         *(mass(k) for k in ('qv','qc','qi','qs')),s('solar_input_qc_tot'),s('solar_input_qi_tot'),*req,
         s('solar_coszen'),state.physics.fields['swdnb'],state.physics.fields['swdnt'],
         *(s('solar_'+k) for k in SOLAR_OUTPUTS),np.int32(cfg.nz+1),np.int32(n),np.int32(0),
         np.int32(cfg.nz-1),np.float32(cfg.dt),np.int32(1),np.int32(cfg.mp_physics!=0),
         np.int32(fqi),np.int32(fqs),*(np.int32(x) for x in radii),np.int32(state.phb.ndim==3)))

def registry_rows(registry):
    registry['parameters']['solar_diagnostics']={'type':'integer','enum':[0,1],'default':0,
        'per_domain':False,'consuming_read':'woof/core/solar_diagnostics.py'}
    registry.setdefault('diagnostic_packages',{})['solar_diagnostics']={'1':{
        'native_source':'WRF v4.7.1 phys/module_diag_solar.F','licence':'WRF public domain',
        'maturity':'supported','consumer':'woof/core/solar_diagnostics.py:update',
        'oracle':'tools/surface_diagnostics_wrf461_oracle',
        'evidence_limit':'Strict real native module replay and seven native cloud-producer branches; three-hour stock go/native comparison, native warm start, checkpoint, both feedback modes and ordinary ensemble. No radiation or cloud observation skill claim; native absent-radius words are preserved.'}}
    registry.setdefault('diagnostic_state',{})['solar']={row[0].lower():{'slot':slot,'restart':True,'history':True}
        for slot,row in HISTORY.items()}
