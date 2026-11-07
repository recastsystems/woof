"""Native AFWA base and thermal indices on resident CUDA columns."""
import numpy as np
BASE_HISTORY={k:(k.upper(),u,d) for k,u,d in (
 ('afwa_mslp','Pa','AFWA Diagnostic: Mean sea level pressure'),
 ('afwa_pwat','kg m-2','AFWA Diagnostic: Precipitable Water'),
 ('afwa_totprecip','mm','AFWA Diagnostic: Total simulation precip'))}
THERMAL_HISTORY={k:(k.upper(),'K',d) for k,d in (
 ('afwa_heatidx','AFWA Diagnostic: Heat index'),('afwa_wchill','AFWA Diagnostic: Wind chill'),
 ('afwa_fits','AFWA Diagnostic: Fighter Index of Thermal Stress'))}
PTYPE_HISTORY={k:(k.upper(),'mm',d) for k,d in (
 ('afwa_rain','AFWA Diagnostic: Rain fall'),('afwa_snow','AFWA Diagnostic: Liq Equiv Snow fall'),
 ('afwa_ice','AFWA Diagnostic: Ice fall'),('afwa_fzra','AFWA Diagnostic: Freezing rain fall'),
 ('afwa_snowfall','AFWA Diagnostic: Snow fall'))}
VIL_HISTORY={'afwa_vil':('VIL','kg m-2','Vertically integrated liquid water'),
             'afwa_radarvil':('RADARVIL','kg m-2','Vertically integrated liquid water from Ze')}
RADAR_HISTORY={f'afwa_{k}':(k.upper(),u,d) for k,u,d in (
 ('refd','dbZ','Derived radar reflectivity'),('refd_com','dbZ','Derived composite radar reflectivity'),
 ('echotop','m','Echo top height from Ze'))}
VIS_HISTORY={k:(k.upper(),u,d) for k,u,d in (
 ('afwa_vis','m','AFWA Diagnostic: Visibility'),('afwa_vis_dust','m','AFWA Diagnostic: Visibility due to dust'),
 ('afwa_vis_alpha','dimensionless','AFWA Diagnostic: Vis alpha Weibull term'))}
ICING_HISTORY={f'afwa_{k}':(k.upper(),u,d) for k,u,d in (
 ('fzlev','m','FREEZING LEVEL'),('icing_lg','kg m-2','TOTAL COLUMN INTEGRATED ICING (>50 um)'),
 ('icing_sm','kg m-2','TOTAL COLUMN INTEGRATED ICING (<50 um)'),
 ('qicing_lg_max','kg kg-1','COLUMN MAX ICING MIXING RATIO (>50 um)'),
 ('qicing_sm_max','kg kg-1','COLUMN MAX ICING MIXING RATIO (<50 um)'),
 ('icingtop','m','TOPMOST ICING LEVEL'),('icingbot','m','BOTTOMMOST ICING LEVEL'))}
ICING_RAW=('afwa_qicing_lg','afwa_qicing_sm')
BUOY_HISTORY={f'afwa_{k}':(f'AFWA_{k.upper()}',u,d) for k,u,d in (
 ('cape','J kg-1','AFWA Diagnostic: Convective Avail Pot Energy'),
 ('cin','J kg-1','AFWA Diagnostic: Convective Inhibition'),
 ('zlfc','m','AFWA Diagnostic: Level of Free Convection'),
 ('plfc','Pa','AFWA Diagnostic: Pressure of LFC'),
 ('lidx','K','AFWA Diagnostic: Surface Lifted Index'),
 ('cape_mu','J kg-1','AFWA Diagnostic: Most unstable CAPE 0-180mb'),
 ('cin_mu','J kg-1','AFWA Diagnostic: Most unstable CIN 0-180mb'))}
BUOY_WORK=('afwa_buoy_w','afwa_buoy_dtv','afwa_buoy_b','afwa_buoy_meta')
SEVERE_HISTORY={k:(k.upper(),u,d) for k,u,d in (
 ('afwa_hail','mm','AFWA Diagnostic: Hail Diameter (Weibull)'),
 ('afwa_llws','m s-1','AFWA Diagnostic: 0-2000 ft wind shear'),
 ('afwa_tornado','m s-1','AFWA Diagnostic: Tornado wind speed (Weibull)'),
 ('tcoli_max','kg m-2','Maximum column integrated ice'),
 ('grpl_flx_max','g m-2 s-1','Maximum graupel flux'))}
from woof.core.nwp_diagnostics import HISTORY as _NWP_HISTORY
SEVERE_HISTORY.update({k:_NWP_HISTORY[k] for k in
                      ('wspd10max','w_up_max','w_dn_max','w_mean')})
SEVERE_HISTORY.update({k:BUOY_HISTORY[k] for k in tuple(BUOY_HISTORY)[:5]})
SEVERE_HISTORY['up_heli_max']=('UP_HELI_MAX','m2 s-2','Maximum updraft helicity')
SEVERE_STATE=('afwa_midrh_min','afwa_midrh_min_old','afwa_tornado_mask','afwa_tornado_dur')
SEVERE_WORK=('afwa_severe_rh','afwa_severe_llws','afwa_severe_mask_prev',
             'afwa_severe_dur_prev','afwa_severe_mu_cape','afwa_severe_mu_cin')


def icing_option(cfg):
    # Registry F_QNC is false for MP5/7/8/WSM5/WSM6/Morrison10.
    # Other QNC schemes are unrecognized by native's icing selector.
    if cfg.mp_physics==16:return 7
    if cfg.mp_physics==40:return 6 if getattr(cfg,'progn',0)>0 else 5
    return 0

PACKAGES=dict(afwa_diag_opt=BASE_HISTORY,afwa_therm_opt=THERMAL_HISTORY,
              afwa_ptype_opt=PTYPE_HISTORY,afwa_vil_opt=VIL_HISTORY,
              afwa_radar_opt=RADAR_HISTORY,afwa_vis_opt=VIS_HISTORY,afwa_icing_opt=ICING_HISTORY,afwa_buoy_opt=BUOY_HISTORY,afwa_severe_opt=SEVERE_HISTORY)
HISTORY={k:v for fields in PACKAGES.values() for k,v in fields.items()}
PERSISTENT=(*HISTORY,'afwa_precip',*ICING_RAW,*SEVERE_STATE)
WORK=('afwa_rho','afwa_dz','afwa_t1','afwa_z1','afwa_zero',
      'afwa_temperature','afwa_z','afwa_u','afwa_v','afwa_ze','afwa_theta','afwa_bad_status',*BUOY_WORK,*SEVERE_WORK)
def needs_profile(cfg):
    return bool(cfg.afwa_severe_opt or cfg.afwa_buoy_opt or cfg.afwa_ptype_opt or cfg.afwa_radar_opt or cfg.afwa_vil_opt or cfg.afwa_vis_opt or (cfg.afwa_icing_opt and icing_option(cfg)))
def active_history(cfg):
    return {k:v for opt,fields in PACKAGES.items() if getattr(cfg,opt) for k,v in fields.items()}
def scratch_shapes(cfg):
    if not cfg.afwa_diag_opt:return {}
    plane=(cfg.ny,cfg.nx)
    names=(*BASE_HISTORY,'afwa_precip','afwa_t1','afwa_z1','afwa_zero')
    result={k:plane for k in names}
    result.update({k:(cfg.nz,*plane) for k in ('afwa_rho','afwa_dz')})
    result.update({k:plane for k in active_history(cfg)})
    if needs_profile(cfg):
        result.update({k:(cfg.nz,*plane) for k in ('afwa_temperature','afwa_z')})
    if cfg.afwa_severe_opt or cfg.afwa_vis_opt or cfg.afwa_bad_data_check:
        result.update({k:(cfg.nz,*plane) for k in ('afwa_u','afwa_v')})
    if cfg.afwa_bad_data_check:
        result['afwa_theta']=(cfg.nz,*plane)
        result['afwa_bad_status']=plane
    if cfg.afwa_buoy_opt or cfg.afwa_severe_opt:
        result.update({k:(2,cfg.nz,*plane) for k in BUOY_WORK[:3]})
        result['afwa_buoy_meta']=(2,6,*plane)
    if cfg.afwa_severe_opt:
        result.update({k:plane for k in (*SEVERE_STATE,*SEVERE_WORK)})
        result['afwa_severe_rh']=(cfg.nz,*plane)
        result.update(uh_diag_col=plane,uh_diag_use=plane)
    if cfg.afwa_icing_opt:result.update({k:(cfg.nz,*plane) for k in ICING_RAW})
    if cfg.afwa_radar_opt or cfg.afwa_vil_opt:result['afwa_ze']=(cfg.nz,*plane)
    return result

def capture_atmosphere(state,cfg,atmosphere):
    if not cfg.afwa_diag_opt:return
    s=state.existing_scratch
    s('afwa_t1')[...]=atmosphere['temperature'][0]
    zi=atmosphere['z_interface']
    s('afwa_z1')[...]=np.float32(.5)*(zi[0]+zi[1])
    if needs_profile(cfg):s('afwa_temperature')[...]=atmosphere['temperature']
    check = bad_check_due(state,cfg)
    if cfg.afwa_severe_opt or cfg.afwa_vis_opt or check:
        s('afwa_u')[...]=atmosphere['u'];s('afwa_v')[...]=atmosphere['v']
    # Active microphysics replaces native th_phy after RK and before its
    # heating limiter. With no microphysics it remains the phy_prep value.
    if check and not cfg.mp_physics:s('afwa_theta')[...]=atmosphere['theta']
    if not cfg.mp_physics:
        s('afwa_rho')[...]=atmosphere['rho'];s('afwa_dz')[...]=atmosphere['dz']
        if needs_profile(cfg):s('afwa_z')[...]=np.float32(.5)*(zi[:-1]+zi[1:])

def prepare_microphysics(state,cfg):
    if not cfg.afwa_diag_opt or not cfg.mp_physics:return
    from woof.core.kernels import get_kernel
    n=cfg.nx*cfg.ny;s=state.existing_scratch
    get_kernel('afwa_diagnostics','afwa_prepare_microphysics')(((n+127)//128,),(128,),
        (state.alt,state.php,state.phb,s('afwa_rho'),s('afwa_dz'),s('afwa_z1'),
         s('afwa_z') if needs_profile(cfg) else s('afwa_rho'),
         np.int32(needs_profile(cfg)),np.int32(cfg.nz),np.int32(n),np.int32(state.phb.ndim==3)))

def output_due(state,cfg):
    # Native CurrTime is the beginning of this step. The alarm's previous
    # ring is its most recent actual crossing; end-of-step history is due.
    epoch=float(getattr(state,'domain_start_offset',0.) or 0.)
    now=float(state.elapsed_seconds)-epoch
    interval=float(cfg.output_interval_s)
    return int(np.floor((now+cfg.dt)/interval))>int(np.floor(now/interval))

def bad_check_due(state,cfg):
    # WRF tests CurrTime == simulationStartTime + dt, literally. Native
    # restart keeps that simulation origin; the run-start clock is distinct.
    return bool(getattr(cfg,'afwa_bad_data_check',0)>0 and output_due(state,cfg)
                and float(state.elapsed_seconds)!=float(cfg.dt))

def capture_microphysics_theta(state,cfg,theta):
    """Native AFWA reads raw post-MP th_phy, even when heating is limited
    or disabled (module_big_step_utilities_em.F:5688-5776)."""
    if getattr(cfg,'afwa_diag_opt',0) and bad_check_due(state,cfg):
        state.existing_scratch('afwa_theta')[...]=theta

def restore_theta_ring(state,cfg,slices):
    """WRF never dispatches microphysics on the specified/nested ring.
    Called after the prognostic ring is restored; native th_phy there is
    the full pre-MP theta from moist_physics_prep_em, not a scheme result.
    """
    if getattr(cfg,'afwa_diag_opt',0) and bad_check_due(state,cfg):
        theta=state.existing_scratch('afwa_theta')
        for part in slices:
            thb=state.thb[part] if state.thb.ndim==3 else state.thb[:,None,None]
            theta[part]=state.thp[part]+thb

def _bad_data(flags):
    raise FloatingPointError('AFWA Diagnostics: model winds and/or potential '
        'temperature appear to be bad (native u/v outside [-300,300] m s-1 '
        'or theta outside [99,9999] K). Set afwa_bad_data_check=0 to disable.')

def check_bad_data(state,cfg):
    if not bad_check_due(state,cfg):return
    from woof.core.kernels import get_kernel
    from woof.core.health_ledger import read_status
    s=state.existing_scratch
    status=s('afwa_bad_status').view('u4').reshape(-1)[:1]
    status.fill(0)
    n=cfg.nx*cfg.ny*cfg.nz
    get_kernel('afwa_diagnostics','afwa_bad_data_check')(((n+255)//256,),(256,),
        (s('afwa_u'),s('afwa_v'),s('afwa_theta'),status,np.int32(n)))
    flags=read_status(status,site='afwa_bad_data_check',describe=_bad_data)
    if flags:_bad_data(flags)

def update(state,cfg):
    if not cfg.afwa_diag_opt:return
    check_bad_data(state,cfg)
    from woof.core.kernels import get_kernel
    n=cfg.nx*cfg.ny;s=state.existing_scratch;z=s('afwa_zero');p=state.physics
    micro=getattr(p,'microphysics',None);fields=getattr(p,'fields',{})
    def amount(owner,name):
        a=getattr(owner,name,None);return z if a is None else a
    due=output_due(state,cfg)
    get_kernel('afwa_diagnostics','afwa_base')(((n+127)//128,),(128,),
        (state.qv,s('afwa_rho'),s('afwa_dz'),
         s('afwa_t1'),s('afwa_z1'),state.ht,fields['psfc'],
         amount(p,'cu_raincv'),amount(micro,'rainncv'),amount(p,'rainc'),amount(micro,'rainnc'),
         s('afwa_precip'),s('afwa_totprecip'),s('afwa_pwat'),s('afwa_mslp'),
         np.int32(cfg.nz),np.int32(n),np.int32(cfg.mp_physics != 0),np.int32(due)))
    if cfg.afwa_severe_opt:
        from woof.core.afwa_severe import update as update_severe
        update_severe(state,cfg,due)
    if cfg.afwa_therm_opt and due:
        get_kernel('afwa_diagnostics','afwa_thermal')(((n+127)//128,),(128,),
            (fields['t2'],fields['psfc'],fields['q2'],fields['u10'],fields['v10'],
             *(s(k) for k in THERMAL_HISTORY),np.int32(n)))
    launch=(((n+127)//128,),(128,));i32=np.int32
    if cfg.afwa_ptype_opt:
        get_kernel('afwa_diagnostics','afwa_ptype')(*launch,
            (s('afwa_temperature'),state.p,state.qv,s('afwa_z'),s('afwa_dz'),state.ht,
             fields['t2'],fields['swdown'],s('afwa_precip'),*(s(k) for k in PTYPE_HISTORY),
             np.float32(cfg.afwa_ptype_ccn_tmp),np.float32(cfg.afwa_ptype_tot_melt),i32(cfg.nz),i32(n)))
    if due and (cfg.afwa_radar_opt or cfg.afwa_vil_opt):
        get_kernel('afwa_diagnostics','afwa_dbz')(((n*cfg.nz+127)//128,),(128,),
            (s('afwa_rho'),s('afwa_temperature'),state.qr,state.qs,state.qg,s('afwa_ze'),i32(n*cfg.nz)))
        get_kernel('afwa_diagnostics','afwa_radar_vil')(*launch,
            (state.qr,state.qs,state.qg,s('afwa_rho'),s('afwa_dz'),s('afwa_z'),s('afwa_ze'),
             *(s(k) if cfg.afwa_vil_opt else z for k in VIL_HISTORY),
             *(s(k) if cfg.afwa_radar_opt else z for k in RADAR_HISTORY),
             i32(cfg.nz),i32(n),i32(cfg.afwa_vil_opt),i32(cfg.afwa_radar_opt)))
    if due and cfg.afwa_vis_opt:
        get_kernel('afwa_diagnostics','afwa_visibility')(*launch,
            (s('afwa_temperature'),state.p,state.qv,s('afwa_z'),s('afwa_u'),s('afwa_v'),state.ht,
             state.qr,state.qs,state.qg,s('afwa_rho'),fields['psfc'],fields['t2'],fields['q2'],
             s('afwa_pwat'),*(s(k) for k in VIS_HISTORY),i32(cfg.nz),i32(n)))

    if due and cfg.afwa_buoy_opt:
        launch_buoyancy((s('afwa_temperature'),state.p,state.qv,s('afwa_z'),state.ht,
                        *(s(k) for k in BUOY_HISTORY),*(s(k) for k in BUOY_WORK),
                        i32(cfg.nz),i32(n)))
    if due and cfg.afwa_icing_opt and icing_option(cfg):
        # Use the actual prognostic cloud number. A constant/artificial
        # droplet field would incorrectly activate native F_QNC=false cases.
        launch_icing(
            (s('afwa_temperature'),s('afwa_z'),s('afwa_dz'),s('afwa_rho'),state.qr,state.qc,state.nc,
             *(s(k) for k in ICING_HISTORY),*(s(k) for k in ICING_RAW),
             i32(cfg.nz),i32(cfg.nz),i32(cfg.nz),i32(icing_option(cfg)),i32(n)))


def launch_icing(args):
    """Parallel native mixing followed by ordered per-column integrals.

    Mixing has no cross-level dependencies. The integration keeps every
    native floating addition in level order; both kernels stay on device.
    """
    from woof.core.kernels import get_kernel
    t,z,dz,rho,qr,qc,number = args[:7]
    qlarge,qsmall = args[14:16]
    nz,levels,native_top,opt,count = args[16:]
    get_kernel('afwa_diagnostics','afwa_icing_mixing')(
        ((int(nz)*int(count)+255)//256,), (256,),
        (t,rho,qr,qc,number,qlarge,qsmall,nz,levels,opt,count))
    get_kernel('afwa_diagnostics','afwa_icing_integrate')(
        ((int(count)+127)//128,), (128,), args)


def launch_buoyancy(args):
    """Native parcel thermodynamics with independent levels on GPU threads."""
    from woof.core.kernels import get_kernel
    t,p,q,z,ht,*tail=args
    outputs=tail[:7];w,d,b,meta,nz,n=tail[7:]
    plane=(((2*int(n)+127)//128,),(128,))
    voxel=(((int(n)*int(nz)+127)//128,),(128,))
    get_kernel('afwa_buoyancy','afwa_buoy_environment')(*voxel,(t,p,q,w,nz,n))
    get_kernel('afwa_buoyancy','afwa_buoy_source')(*plane,(t,p,q,w,meta,nz,n))
    get_kernel('afwa_buoyancy','afwa_buoy_lift')(((2*int(n)*int(nz)+127)//128,),(128,),
        (t,p,w,meta,d,b,nz,n))
    get_kernel('afwa_buoyancy','afwa_buoy_integrate')(*plane,(p,z,ht,meta,d,b,*outputs,nz,n))
