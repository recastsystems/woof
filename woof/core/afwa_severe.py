"""WRF AFWA interval severe diagnostics; all field work stays on device."""
import numpy as np


def active_species(cfg):
    # Registry.EM_COMMON and registry.sbm F_QNG, plus all six mass species.
    return cfg.mp_physics in (9,10,18,30,32,38,40,56)


def surface_buoyancy(args):
    from woof.core.kernels import get_kernel
    t,p,q,z,ht,*tail=args
    outputs=tail[:7];w,d,b,meta,nz,n=tail[7:]
    plane=(((int(n)+127)//128,),(128,))
    voxel=(((int(n)*int(nz)+127)//128,),(128,))
    get_kernel('afwa_buoyancy','afwa_buoy_environment')(*voxel,(t,p,q,w,nz,n))
    get_kernel('afwa_buoyancy','afwa_severe_buoy_source')(*plane,(t,p,q,w,meta,nz,n))
    get_kernel('afwa_buoyancy','afwa_severe_buoy_lift')(*voxel,(t,p,w,meta,d,b,nz,n))
    get_kernel('afwa_buoyancy','afwa_severe_buoy_integrate')(*plane,(p,z,ht,meta,d,b,*outputs,nz,n))


def update(state,cfg,due):
    from woof.core.kernels import get_kernel
    from woof.core.afwa_diagnostics import BUOY_HISTORY,BUOY_WORK
    s=state.existing_scratch;f=state.physics.fields;n=cfg.nx*cfg.ny
    plane=(((n+127)//128,),(128,));i32=np.int32
    epoch=float(getattr(state,'domain_start_offset',0.) or 0.)
    now=float(state.elapsed_seconds)-epoch
    reset=now % float(cfg.output_interval_s) < float(cfg.dt)
    get_kernel('afwa_severe','afwa_severe_driver')(*plane,
        (s('afwa_u'),s('afwa_v'),s('afwa_z'),state.ht,f['u10'],f['v10'],s('afwa_precip'),
         *(s(k) for k in ('wspd10max','afwa_llws','afwa_severe_llws','w_up_max','w_dn_max',
                          'tcoli_max','grpl_flx_max','up_heli_max','afwa_tornado',
                          'afwa_midrh_min_old','afwa_midrh_min','afwa_hail')),
         i32(reset),np.float32(cfg.dt),i32(cfg.nz),i32(n)))
    if not active_species(cfg):return
    from woof import wrf_exact
    if wrf_exact.ENABLED:
        get_kernel('afwa_severe','afwa_severe_humidity')(((n*cfg.nz+255)//256,),(256,),
            (state.p,s('afwa_temperature'),state.qv,s('afwa_severe_rh'),i32(n*cfg.nz)))
    get_kernel('afwa_severe','afwa_severe_column')(*plane,
        (state.w,s('afwa_temperature'),s('afwa_z'),state.p,
         state.qi,state.qs,state.qg,s('afwa_rho'),s('afwa_dz'),s('afwa_severe_rh'),
         state.ht,f['t2'],s('uh_diag_col'),s('afwa_midrh_min_old'),s('afwa_midrh_min'),
         s('w_up_max'),s('w_dn_max'),s('afwa_hail'),s('tcoli_max'),s('grpl_flx_max'),i32(cfg.nz),i32(n),state.qv,i32(not wrf_exact.ENABLED)))
    if due or float(state.elapsed_seconds)==float(cfg.dt):
        surface_buoyancy((s('afwa_temperature'),state.p,state.qv,s('afwa_z'),state.ht,
            *(s(k) for k in tuple(BUOY_HISTORY)[:5]),s('afwa_severe_mu_cape'),s('afwa_severe_mu_cin'),
            *(s(k) for k in BUOY_WORK),i32(cfg.nz),i32(n)))
    s('afwa_severe_mask_prev')[...]=s('afwa_tornado_mask')
    s('afwa_severe_dur_prev')[...]=s('afwa_tornado_dur')
    # Native specified/nested west/south start at ids+1/jds+1. The east
    # and north end at ide-1/jde-1, the last stored mass point.
    west=int(cfg.specified or cfg.nested or cfg.open_x)
    south=int(cfg.specified or cfg.nested or cfg.open_y)
    get_kernel('afwa_severe','afwa_severe_tornado')(*plane,
        (s('uh_diag_col'),s('afwa_severe_llws'),s('afwa_zlfc'),s('afwa_midrh_min'),s('afwa_midrh_min_old'),
         s('afwa_severe_mask_prev'),s('afwa_severe_dur_prev'),s('afwa_tornado_mask'),s('afwa_tornado_dur'),
         s('afwa_tornado'),np.float32(cfg.dt),i32(cfg.ny),i32(cfg.nx),i32(west),i32(0),i32(south),i32(0),i32(not (cfg.open_x or cfg.specified or cfg.nested)),i32(not (cfg.open_y or cfg.specified or cfg.nested))))
