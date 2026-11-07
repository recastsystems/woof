"""Bind native CAM driver fields to the model's existing physics slots."""
from __future__ import annotations
import cupy as cp
import numpy as np
from woof.core.cam5_coupling import DEEP_EXPORTS,LIQUID_NUMBER,field_shapes,selected
from woof.core.cam5_workspace import CamWorkspace


def initialize_fields(fields,cfg):
    from woof.core.cam5_memory import CAMZM_INT_OUTPUTS
    for name,shape in field_shapes(cfg).items():
        if name not in fields:
            fields[name]=cp.zeros(shape,dtype=cp.int32 if name in CAMZM_INT_OUTPUTS else cp.float32)


def workspace(driver):
    value=getattr(driver,'_cam_workspace',None)
    if value is None:
        driver._cam_workspace=value=CamWorkspace()
    return value


def stable_inputs(owner,prefix,inputs):
    # Geometry expressions may allocate new arrays. Kernels with pointer tables
    # consume stable owned addresses, updated device-to-device on this stream.
    result={}
    for name,value in inputs.items():
        target=owner.array(prefix+name,value.shape,value.dtype)
        cp.copyto(target,value);result[name]=target
    return result


def couple_numbers(driver,state,cfg,rates):
    from woof.core.physics import _specified_mass_mask
    mass=state.c1h[:,None,None]*state.total_mu()[None]+state.c2h[:,None,None]
    result={}
    for name,rate in rates.items():
        if getattr(state,name,None) is None:
            raise ValueError(f'CAM number tendency {name} has no state carrier')
        value=mass*rate
        if cfg.specified or cfg.nested:_specified_mass_mask(value)
        result[name]=value
    return result


def cu_number_rates(driver,state,cfg):
    from woof.config import UW_PBL_ICE_NUMBER_SPECIES
    result={}
    for name,rate in ((LIQUID_NUMBER.get(cfg.mp_physics),'rqcncuten'),
                       (UW_PBL_ICE_NUMBER_SPECIES.get(cfg.mp_physics),'rqincuten')):
        if name is not None:result[name]=driver.fields['cam_held_'+rate]
    return result


class CAMDeepConvection:
    def bind_driver(self,driver):self._driver=driver
    def release(self):self._driver=None;return 0
    def __call__(self,*,atmosphere,fields,state,cfg):
        from woof.core.camzm import camzm_step
        from woof.core.physics import CumulusResult
        f=fields;a=atmosphere;zi=a['z_interface'];nz,ny,nx=state.p.shape
        owner=workspace(self._driver)
        if cfg.bl_pbl_physics == 2:
            f['tke_pbl'][:-1] = f['tke_myj']
            f['tke_pbl'][-1].fill(0)
        cldfra=(state.cldfra_mp_all if cfg.mp_physics==11 else self._driver.cldfra)
        if cldfra is None:cldfra=owner.array('cu7/zero_cloud',state.p.shape,cp.float32,clear=True)
        inputs=dict(th=a['theta'],t=a['temperature'],p=a['pressure'],pi=a['exner'],
            qv=a['qv'],qc=a['qc'],qi=a['qi'],z=np.float32(.5)*(zi[:-1]+zi[1:]),
            dz8w=a['dz'],u=a['u'],v=a['v'],cldfra=cldfra,
            p8w=a['p_interface'],z_at_w=zi,tke_pbl=f['tke_pbl'],
            ht=cp.broadcast_to(cp.asarray(state.ht,dtype=cp.float32),(ny,nx)),
            psfc=a['p_interface'][0],tpert=f['tpert2d'] if cfg.bl_pbl_physics==9 else owner.array('cu7/zero_surface',(ny,nx),cp.float32,clear=True),kpbl=f['kpbl'])
        for n in ('tsk','ust','mavail','pblh','xland','hfx','qfx'):inputs[n]=f[n]
        owner=workspace(self._driver)
        inputs=stable_inputs(owner,'cu7/input/',inputs)
        epoch=float(getattr(state,'domain_start_offset',0.) or 0.)
        step=int(np.floor((float(state.elapsed_seconds)-epoch)/cfg.dt+.5))+1
        out=camzm_step(inputs,dt=cfg.dt,dx=cfg.dx,cudt=cfg.cudt_minutes,
            itimestep=step,bl_pbl_physics=cfg.bl_pbl_physics,
            sf_sfclay_physics=cfg.sf_sfclay_physics,
            diagnostics=True,workspace=owner)
        for n in DEEP_EXPORTS:f[n][...]=out[n]
        f['icwmrdp'][...]=out['icwmrdp3d']
        for name,value in out.items():
            if name in f and name not in DEEP_EXPORTS:
                target=f[name]
                if target.shape == value.shape:target[...]=value
                elif target.shape[0] == value.shape[0]+1:
                    target[:-1]=value;target[-1].fill(0)
                else:raise ValueError(f'CAM diagnostic {name} shape mismatch')
        for n in ('rucuten','rvcuten','rqcncuten','rqincuten'):
            f['cam_held_'+n][...]=out[n]
        # advance_ppt accumulates the native rain rate on every model step,
        # including steps skipped by a positive CUDT. No interval increment.
        return CumulusResult(rthcuten=out['rthcuten'],rqvcuten=out['rqvcuten'],
            rqccuten=out['rqccuten'],
            rqicuten=out['rqicuten'] if getattr(state,'qi',None) is not None else None,
            rucuten=out['rucuten'],rvcuten=out['rvcuten'],pratec=out['pratec'],
            persistent_pratec=True,scalar_tendencies=cu_number_rates(self._driver,state,cfg))


def before_compose(driver,atmosphere,state,cfg):
    if not selected(cfg):return
    f=driver.fields
    if cfg.bl_pbl_physics == 2:
        f['tke_pbl'][:-1] = f['tke_myj']
        f['tke_pbl'][-1].fill(0)
    if cfg.mp_physics==11:
        f['cam_t_phy'][...]=atmosphere['temperature']
        f['cam_p_hyd'][...]=atmosphere['pressure']
        f['cam_p8w'][...]=atmosphere['p_interface']
    if cfg.bl_pbl_physics != 9:
        f['cam_kvh'][:-1]=f['exch_h'];f['cam_kvh'][-1].fill(0)
    if getattr(cfg,'shcu_physics',0)!=2:return
    from woof.core.uwshcu import uwshcu_step
    from woof.config import UW_PBL_ICE_NUMBER_SPECIES
    from woof.core.physics import couple_column_tendencies
    a=atmosphere;zi=a['z_interface'];owner=workspace(driver)
    zero=owner.array('shcu2/zero',state.p.shape,cp.float32,clear=True)
    liquid=LIQUID_NUMBER.get(cfg.mp_physics);ice=UW_PBL_ICE_NUMBER_SPECIES.get(cfg.mp_physics)
    qnc=getattr(state,liquid,None) if liquid else None
    qni=getattr(state,ice,None) if ice else None
    inputs=dict(u=a['u'],v=a['v'],t=a['temperature'],qv=a['qv'],qc=a['qc'],qi=a['qi'],
        qnc=qnc if qnc is not None else zero,qni=qni if qni is not None else zero,
        p=a['pressure'],z=np.float32(.5)*(zi[:-1]+zi[1:]),pi=a['exner'],
        p8w=a['p_interface'],z_at_w=zi,tke_pbl=f['tke_pbl'],pblh=f['pblh'],
        ht=cp.broadcast_to(cp.asarray(state.ht,dtype=cp.float32),state.p.shape[1:]),
        dlf_in=f['dlf'],cmfmc_in=f['cmfmc'],cutop_in=f['cutop'],cubot_in=f['cubot'],rliq_in=f['rliq'])
    inputs=stable_inputs(owner,'shcu2/input/',inputs)
    out=uwshcu_step(inputs,{'cush':f['cush']},dt=cfg.dt,has_qnc=qnc is not None,
        has_qni=qni is not None,workspace=owner)
    for n,value in out.items():f[n][...]=value
    driver.shallow_tendencies=couple_column_tendencies(state,cfg,ru=out['rushten'],
        rv=out['rvshten'],rtheta=out['rthshten'],rqv=out['rqvshten'],rqc=out['rqcshten'],
        rqi=out['rqishten'] if getattr(state,'qi',None) is not None else None)
    rates={}
    if qnc is not None:rates[liquid]=out['rqcnshten']
    if qni is not None:rates[ice]=out['rqinshten']
    driver.shallow_tendencies.extra_scalars=couple_numbers(driver,state,cfg,rates)
    increment=out['pratesh']*np.float32(cfg.dt)
    f['rainsh']+=increment
    driver._pending_rainbl+=cp.maximum(increment,np.float32(0.))
    if driver.ruc_params is not None or driver.noahmp_params is not None:
        f['surface_raincv']+=cp.maximum(increment,np.float32(0.))
    driver.call_counts['shallow_cumulus']=driver.call_counts.get('shallow_cumulus',0)+1
