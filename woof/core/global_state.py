"""Bind native spherical fields to the ordinary domain and restart owner."""
import numpy as np


def validate_config(cfg):
    """Direct domain callers obey the same equations as the stock importer."""
    expected=dict(polar=True,moist=False,nested=False,specified=False,
        open_x=False,open_y=False,terrain_opt=0,mp_physics=0,
        ra_sw_physics=0,sf_surface_physics=0,sf_sfclay_physics=0,
        bl_pbl_physics=0,cu_physics=0,diff_opt=1,km_opt=1,khdif=0.,
        kvdif=0.,damp_opt=0,diff_6th_opt=0,w_damping=0)
    for name,value in expected.items():
        if getattr(cfg,name)!=value:
            raise ValueError(f'global {name}={getattr(cfg,name)!r} has no binding in the dry spherical driver; expected {value!r}')
    if cfg.ra_lw_physics not in (0,31):
        raise ValueError('dry spherical driver requires ra_lw_physics=0 or 31')
    if min(cfg.nx,cfg.ny)<7 or cfg.nz<4:
        raise ValueError('global fifth-order fluxes require nx/ny>=7,nz>=4')
    if not np.isfinite(cfg.fft_filter_lat) or not 0<=cfg.fft_filter_lat<=90:
        raise ValueError('global fft_filter_lat must be in [0,90]')


def bind(state, cfg, raw):
    """Cold-start only; subsequent work uses domain-owned stable pointers."""
    import cupy as cp
    validate_config(cfg)
    aliases=dict(t_2='thp',mu_2='mup',ph_2='php',p='p_perturbation',
        msftx='global_tx',msfty='msft',msfux='global_ux',msfuy='msfu',
        msfvx='msfv',msfvy='global_vy',msfvx_inv='msfv_inv',
        xlat='global_lat',xlong='global_lon',mub='mub2d')
    for source,target in aliases.items():
        if source in raw:
            value=cp.asarray(raw[source],dtype=cp.float32,order='C')
            current=getattr(state,target,None)
            if current is None:
                setattr(state,target,value)
            else:
                if current.shape!=value.shape:
                    raise ValueError(f'global {source} shape {value.shape} differs from {current.shape}')
                current[...]=value
    for name in ('u','v','w','pb','alb','phb','al'):
        if name in raw:getattr(state,name)[...]=cp.asarray(raw[name],dtype=cp.float32)
    for name in ('znw','znu','dnw','rdnw','dn','rdn','fnm','fnp',
                 'c1h','c2h','c3h','c4h','c1f','c2f','c3f','c4f'):
        target=getattr(state,name)
        target[...]=cp.asarray(raw[name][:target.size],dtype=cp.float32)
    if not hasattr(state,'global_ux'):
        a=cp.empty((cfg.ny,cfg.nx+1),cp.float32)
        a[:,:cfg.nx]=state.global_tx;a[:,cfg.nx]=state.global_tx[:,0]
        state.global_ux=a
    state.msfu.fill(1)
    state.f[...]=cp.asarray(raw['f']);state.e[...]=cp.asarray(raw['e'])
    state.thb.fill(300);state.cosa.fill(1);state.sina.fill(0)
    scalars=cp.asnumpy(cp.asarray(raw['scalars'],dtype=cp.float32))
    state.p_top=float(scalars[0])
    state.mub=np.float32(cp.asnumpy(state.mub2d)[0,0])
    for name,value in zip(('cf1','cf2','cf3','cfn','cfn1'),scalars[3:]):
        setattr(state,name,np.float32(value))
    state.global_rthraten=cp.zeros((cfg.nz,cfg.ny,cfg.nx),cp.float32)
    state.has_msf=True;state.rotational=True
    cp.add(state.al,state.alb,out=state.alt)
    cp.add(state.p_perturbation,state.pb,out=state.p)
    return state


def build_stepper(state,cfg):
    validate_config(cfg)
    import cupy as cp
    from woof.core.global_step import GlobalHydrostaticStepper
    fields=dict(u=state.u,v=state.v,w=state.w,t=state.thp,ph=state.php,
        mu=state.mup,p=state.p_perturbation,al=state.al,alt=state.alt,pressure=state.p,
        mub=state.mub2d,pb=state.pb,alb=state.alb,phb=state.phb,
        lat=state.global_lat,tx=state.global_tx,ty=state.msft,
        ux=state.global_ux,uy=state.msfu,vx=state.msfv,vy=state.global_vy,
        vxinv=state.msfv_inv,f=state.f,e=state.e,rthraten=state.global_rthraten)
    for name in ('znw','znu','dnw','rdnw','dn','rdn','fnm','fnp',
                 'c1h','c2h','c3h','c4h','c1f','c2f','c3f','c4f'):
        value=getattr(state,name)
        padded=state.scratch((cfg.nz+1,), 'global_rk_padded_'+name)
        padded.fill(0);padded[:value.size]=value;fields[name]=padded
    fields['scalars']=state.scratch((8,), 'global_rk_scalars')
    fields['scalars'][...]=cp.asarray((state.p_top,cfg.dx,cfg.dy,state.cf1,state.cf2,
                                     state.cf3,state.cfn,state.cfn1),cp.float32)
    stepper=GlobalHydrostaticStepper(fields,dt=cfg.dt,sound_steps=cfg.time_step_sound,
        filter_lat=cfg.fft_filter_lat,smdiv=cfg.smdiv,emdiv=cfg.emdiv,epssm=cfg.epssm,
        hypsometric_opt=cfg.hypsometric_opt,v_mom_order=cfg.v_mom_adv_order,
        v_sca_order=cfg.v_sca_adv_order,lw=cfg.ra_lw_physics,owner=state)
    stepper.capture()
    state._global_stepper=stepper
    return stepper


def advance(state,cfg):
    stepper=getattr(state,'_global_stepper',None)
    if stepper is None:stepper=build_stepper(state,cfg)
    stepper.step()
    state.elapsed_seconds+=cfg.dt
