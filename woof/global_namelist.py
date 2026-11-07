"""Stock WRF admission for dry hydrostatic spherical integration."""
from datetime import datetime
import math
from pathlib import Path
from woof.config import RunConfig
from woof.fortran_namelist import parse_namelist


def is_global(document):
    return document.get('bdy_control',{}).get('polar',[False])[0] is True


def import_global_namelist(path):
    document=parse_namelist(path)
    # Every admitted setting is either bound below or inert for this dry
    # fixed-step topology. Unknown active knobs cannot silently disappear.
    supported={
      'time_control': 'run_days run_hours run_minutes run_seconds start_year start_month start_day start_hour start_minute start_second end_year end_month end_day end_hour end_minute end_second history_interval frames_per_outfile restart restart_interval io_form_history io_form_restart io_form_input io_form_boundary',
      'domains': 'time_step time_step_fract_num time_step_fract_den max_dom s_we e_we s_sn e_sn s_vert e_vert dx dy ztop grid_id parent_id i_parent_start j_parent_start parent_grid_ratio parent_time_step_ratio feedback smooth_option',
      'physics': 'mp_physics ra_lw_physics ra_sw_physics radt sf_sfclay_physics sf_surface_physics bl_pbl_physics bldt cu_physics cudt',
      'fdda': 'grid_fdda grid_sfdda obs_nudge_opt',
      'dynamics': 'hybrid_opt etac rk_ord diff_opt km_opt damp_opt zdamp dampcoef khdif kvdif smdiv emdiv epssm time_step_sound h_mom_adv_order v_mom_adv_order h_sca_adv_order v_sca_adv_order fft_filter_lat non_hydrostatic diff_6th_opt w_damping hypsometric_opt phi_adv_z',
      'bdy_control': 'polar periodic_x periodic_y specified nested open_xs open_xe open_ys open_ye symmetric_xs symmetric_xe symmetric_ys symmetric_ye',
      'grib2': '', 'namelist_quilt': 'nio_tasks_per_group nio_groups'}
    for section,entries in document.items():
        for key in entries:
            if key not in supported.get(section,'').split():
                raise ValueError(f'&{section} {key} has no spherical operator or IO binding; ignoring it would change the requested experiment')
    def get(section,key,default=None):
        return document.get(section,{}).get(key,[default])[0]
    def require(section,key,value,default=None):
        actual=get(section,key,default)
        if actual!=value or isinstance(actual,bool)!=isinstance(value,bool):
            raise ValueError(f'&{section} {key}={actual!r}: the resident global driver requires {value!r}; this setting selects a different topology or an unbound global operator')
    require('domains','max_dom',1,1)
    require('physics','radt',0,0)
    require('dynamics','phi_adv_z',1,1)
    frames=get('time_control','frames_per_outfile',1)
    if type(frames)is not int or frames<1:
        raise ValueError('frames_per_outfile must be a positive integer')
    for key in ('io_form_history','io_form_restart','io_form_input','io_form_boundary'):
        require('time_control',key,2,2)
    require('namelist_quilt','nio_tasks_per_group',0,0)
    require('namelist_quilt','nio_groups',1,1)
    require('bdy_control','polar',True,False)
    require('bdy_control','periodic_x',True,False)
    for name in ('periodic_y','specified','nested','open_xs','open_xe','open_ys','open_ye',
                 'symmetric_xs','symmetric_xe','symmetric_ys','symmetric_ye'):
        require('bdy_control',name,False,False)
    require('dynamics','non_hydrostatic',False,True)
    for name,value,default in [('rk_ord',3,3),('diff_opt',1,1),('km_opt',1,1),('damp_opt',0,0),
                               ('khdif',0.,0.),('kvdif',0.,0.),('diff_6th_opt',0,0),
                               ('h_mom_adv_order',5,5),('h_sca_adv_order',5,5),('w_damping',0,0)]:
        actual=get('dynamics',name,default)
        if isinstance(actual,bool) or actual!=value:
            raise ValueError(f'&dynamics {name}={actual}: its global operator is not bound; dropping it would change the requested equations')
    for name in ('mp_physics','ra_sw_physics','sf_sfclay_physics','sf_surface_physics','bl_pbl_physics','cu_physics'):
        require('physics',name,0,0)
    for name in ('grid_fdda','grid_sfdda','obs_nudge_opt'):
        require('fdda',name,0,0)
    lw=get('physics','ra_lw_physics',0)
    if isinstance(lw,bool) or lw not in (0,31):
        raise ValueError(f'ra_lw_physics={lw}: spherical column binding currently supplies options 0 and 31')
    nz=get('domains','e_vert')-get('domains','s_vert',1)
    ny=get('domains','e_sn')-get('domains','s_sn',1)
    nx=get('domains','e_we')-get('domains','s_we',1)
    if nx<7 or ny<7 or nz<4:raise ValueError('native fifth-order global fluxes require nx/ny>=7,nz>=4')
    dt=get('domains','time_step',0)+get('domains','time_step_fract_num',0)/get('domains','time_step_fract_den',1)
    start=datetime(*(get('time_control','start_'+key,default)for key,default in
                     [('year',1),('month',1),('day',1),('hour',0),('minute',0),('second',0)]))
    duration=sum(get('time_control','run_'+key,0)*factor for key,factor in
                 [('days',86400),('hours',3600),('minutes',60),('seconds',1)])
    if not duration:
        end=datetime(*(get('time_control','end_'+key,default)for key,default in
                       [('year',start.year),('month',start.month),('day',start.day),('hour',start.hour),('minute',0),('second',0)]))
        duration=(end-start).total_seconds()
    history=get('time_control','history_interval',60)*60
    restart=get('time_control','restart_interval',0)*60
    sound=get('dynamics','time_step_sound',4)
    for name,value in [('dt',dt),('duration',duration),('history',history)]:
        if not isinstance(value,(int,float))or not math.isfinite(value)or value<=0:raise ValueError(f'global {name} must be finite and positive')
    for name,value in [('duration',duration),('history',history),('restart',restart)]:
        if value and abs(value/dt-round(value/dt))>1e-8:raise ValueError(f'global {name} must land on a fixed timestep')
    if type(sound)is not int or sound<2 or sound%2:raise ValueError('global acoustic count must be even and >=2')
    for name in ('v_mom_adv_order','v_sca_adv_order'):
        if get('dynamics',name,3)not in (3,5):raise ValueError(f'{name} must be 3 or 5 for the native global vertical flux')
    fflat=get('dynamics','fft_filter_lat',45.)
    if not 0<=fflat<=90:raise ValueError('fft_filter_lat must be in [0,90]')
    cfg=RunConfig(nx=nx,ny=ny,nz=nz,dx=get('domains','dx'),dy=get('domains','dy'),
        ztop=get('domains','ztop'),dt=dt,run_seconds=duration,polar=True,
        ra_lw_physics=lw,ra_sw_physics=0,hybrid_opt=get('dynamics','hybrid_opt',0),
        etac=get('dynamics','etac',.2),time_step_sound=sound,smdiv=get('dynamics','smdiv',.1),
        emdiv=get('dynamics','emdiv',.01),epssm=get('dynamics','epssm',.1),
        h_sca_adv_order=5,v_mom_adv_order=get('dynamics','v_mom_adv_order',3),
        v_sca_adv_order=get('dynamics','v_sca_adv_order',3),fft_filter_lat=fflat,
        diff_opt=1,output_interval_s=history,restart_interval_s=restart,hypsometric_opt=get('dynamics','hypsometric_opt',1))
    if cfg.hybrid_opt not in (0,1,2,3):raise ValueError('global hybrid_opt must be 0..3')
    if not math.isfinite(cfg.ztop)or cfg.ztop<=0:raise ValueError('global ztop must be finite and positive')
    if cfg.hypsometric_opt not in (1,2):raise ValueError('global hypsometric_opt must be 1 or 2')
    # Retain every supplied setting, including unused array elements beyond
    # max_dom, in the import receipt. None becomes a different physics arm.
    return cfg,dict(schema='gpuwm-native-global-import-v1',start=start.isoformat(),
        frames_per_outfile=frames,
        namelist=str(Path(path).resolve()),document=document,
        initialization='native wrfinput when present; public WRF analytic state otherwise',
        dx_dy='native initializer computes spherical spacing from nx/ny',
        observations='ideal year-1 atmosphere has no coincident observations')
