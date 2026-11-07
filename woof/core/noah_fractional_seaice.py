"""WRF 4.7.1 surface-driver fractional ice seam for Noah and Noah-MP.

The threshold, temperature diagnosis and wait-for-LSM fields follow
phys/module_surface_driver.F:1491-1495, 2610-2640, 2933-2996,
3086-3117, 3277-3340 and 7127-7206. No fractional field is rounded to a
whole-cell classification.
"""
import numpy as np


def active(cfg):
    return int(cfg.sf_surface_physics) in (2, 4) and bool(cfg.fractional_seaice)


def prescribed(cfg):
    return int(cfg.sf_surface_physics) in (2, 4) and (
        active(cfg) or cfg.seaice_albedo_opt or cfg.seaice_thickness_opt
        or cfg.seaice_snowdepth_opt or cfg.seaice_albedo_default != .65
        or cfg.seaice_thickness_default != 3.
        or cfg.seaice_snowdepth_min != .001
        or cfg.seaice_snowdepth_max != 1.e10)


def carrier_names(cfg):
    from woof.core.physics_inventory import (SFCLAY_OUTPUTS,
        MYNN_SURFACE_OUTPUTS, MYJ_SFCLAY_INOUT, MYJ_SFCLAY_OUTPUTS)
    names = set(SFCLAY_OUTPUTS)
    if int(cfg.sf_sfclay_physics) == 5:
        names.update(MYNN_SURFACE_OUTPUTS)
    if int(cfg.sf_sfclay_physics) in (2, 4):
        names.update(MYJ_SFCLAY_INOUT)
        names.update('br' if n == 'rib' else n for n in MYJ_SFCLAY_OUTPUTS)
    return tuple(sorted(names))


def ice_temperature(f, itimestep, *, surface=True):
    """get_local_ice_tsk, including WRF's first-three-step SST ladder."""
    import cupy as cp
    F = np.float32
    x = f['xice']; mask = (x >= F(.02)) & (x <= F(1))
    sst = cp.maximum(f['sst'], F(271.4))
    if surface and itimestep <= 3:
        warm = sst > F(273)
        sst = cp.where(warm & (x >= F(.6)), F(271.4),
            cp.where(warm & (x >= F(.4)), F(273),
            cp.where(warm & (x >= F(.2)) & (sst > F(275)), F(275),
            cp.where(warm & (sst > F(278)), F(278), sst))))
    f['sst'][...] = cp.where(mask, sst, f['sst'])
    f['tsk_sea'][...] = cp.where(mask, sst, f['tsk'])
    ice = cp.maximum((f['tsk'] - (F(1)-x)*sst) / cp.where(mask,x,F(1)), F(221.4))
    ice = cp.where((x < F(.2)) & (f['tsk'] < F(253.15)), F(253.15), ice)
    ice = cp.where((x < F(.1)) & (f['tsk'] < F(263.15)), F(263.15), ice)
    return cp.where(mask, ice, f['tsk']).astype(F)


def blend(f, names):
    import cupy as cp
    x=f['xice']; F=np.float32; mask=(x>=F(.02)) & (x<=F(1))
    for n in names:
        f[n][...] = cp.where(mask, f[n]*x+(F(1)-x)*f[n+'_sea'], f[n])


def before_lsm(f, itimestep, surface_due=True):
    import cupy as cp
    F=np.float32; x=f['xice']; mask=(x>=F(.02)) & (x<=F(1)); denominator=cp.where(mask,x,F(1))
    for n,sea in (('albedo',.08),('emiss',.98)):
        f[n][...] = cp.where(mask,(f[n]-(F(1)-x)*F(sea))/denominator,f[n])
    if not surface_due:
        f['tsk'][...] = ice_temperature(f,itimestep,surface=False)


def after_lsm(f, surface_due=True):
    import cupy as cp
    F=np.float32; x=f['xice']; mask=(x>=F(.02)) & (x<=F(1))
    for n,sea in (('albedo',.08),('emiss',.98)):
        f[n][...] = cp.where(mask,f[n]*x+(F(1)-x)*F(sea),f[n])
    if surface_due:
        blend(f, ('flhc','flqc','cpm','cqs2','chs2','chs','qsfc','qgh',
                  *(('qz0',) if 'qz0' in f else ()), 'hfx','qfx','lh'))
        f['tsk_save'][...] = cp.where(mask,f['tsk'],f['tsk_save'])
    f['tsk'][...] = cp.where(mask,f['tsk']*x+(F(1)-x)*f['tsk_sea'],f['tsk'])


def mm5(driver, atmosphere, cfg):
    """The native SFCLAY and SFCLAYREV two-call wrapper; option off never enters."""
    import cupy as cp
    from woof.core.sfclay import launch_sfclay, SFClayResult
    from woof.core.physics_inventory import SFCLAY_OUTPUTS
    f=driver.fields; F=np.float32
    step=int(np.floor(float(driver.state.elapsed_seconds)/cfg.dt+.5))+1
    local=ice_temperature(f,step)
    mask=(f['xice']>=F(.02)) & (f['xice']<=F(1))
    hold_regime=f['regime'].copy()
    for n in SFCLAY_OUTPUTS: f[n+'_sea'][...] = f[n]
    def call(tsk, xland, mavail, result):
        launch_sfclay(atmosphere['u'][0],atmosphere['v'][0],
            atmosphere['temperature'][0],atmosphere['qv'][0],
            atmosphere['pressure'][0],atmosphere['dz'][0],
            atmosphere['p_interface'][0],tsk,f['pblh'],mavail,xland,
            f['lakemask'],result,option=int(cfg.sf_sfclay_physics),dx=cfg.dx,
            dx_field=f.get('dx2d') if int(cfg.sf_sfclay_physics)==1 else None,
            isfflx=bool(cfg.isfflx),isftcflx=cfg.isftcflx,iz0tlnd=cfg.iz0tlnd)
    call(local,f['xland'],f['mavail'],driver.sfclay_result)
    # WRF restores ZNT_HOLD after assigning its temporary 0.0001 value.
    result=SFClayResult(*(f[n+'_sea'] for n in SFCLAY_OUTPUTS))
    call(f['tsk_sea'],cp.where(mask,F(2),f['xland']),
         cp.where(mask,F(1),f['mavail']),result)
    blend(f,('br','gz1oz0','mol','psih','psim','fh','fm','rmol','ust','wspd',
             'zol','cd','cda','ck','cka','q2','t2','th2','u10','ustm','v10'))
    f['regime'][...] = cp.where(mask & (f['xice']>=F(.5)),hold_regime,f['regime'])


def eta(driver, columns, surface, state, outputs, cfg, itimestep):
    """MYJSFC/QNSESFC_SEAICE_WRAPPER (surface_driver.F:4508-5240)."""
    import cupy as cp
    from woof.core.myjsfc import launch_myj_sfclay
    from woof.core.kernels import get_kernel
    f=driver.fields; F=np.float32
    mask=(f['xice']>=F(.02)) & (f['xice']<=F(1))
    f['tsk'][...] = ice_temperature(f,itimestep)
    n=f['tsk'].size
    def qsat(tsk):
        out=cp.empty_like(tsk)
        get_kernel('noah_fractional','eta_ice_qsat')(((n+127)//128,),(128,),
                 (tsk,surface['psfc'],out,np.int32(n)))
        return out
    qs=qsat(f['tsk'])
    f['qsfc'][...] = cp.where(mask,(f['qsfc']-(F(1)-f['xice'])*qs)/cp.where(mask,f['xice'],F(1)),f['qsfc'])
    for name in carrier_names(cfg):
        if name in f:f[name+'_sea'][...] = f[name]
    def call(sf,st,out):
        launch_myj_sfclay(columns,sf,st,out,itimestep=itimestep,
                          sf_sfclay_physics=int(cfg.sf_sfclay_physics),
                          ocean_z0=driver.ocean.z0 if getattr(driver,'ocean',None) is not None else None)
    call(surface,state,outputs)
    # WRF restores ZNT_HOLD after assigning its temporary 0.0001 value.
    f['qsfc_sea'][...] = cp.where(mask,qsat(f['tsk_sea']),f['qsfc_sea'])
    sea_surface={**surface,'tsk':f['tsk_sea'],
       'xland':cp.where(mask,F(2),f['xland']),
       'mavail':cp.where(mask,F(1),f['mavail']),
       'z0base':cp.where(mask,F(.0001),f['z0'])}
    call(sea_surface,{n:f[n+'_sea'] for n in state},
         {n:f[('br' if n=='rib' else n)+'_sea'] for n in outputs})
    blend(f,('ct','pshltr','qshltr','q2','q10','th2','th10','tshltr','t2',
             'u10','v10','thz0','uz0','vz0','ust','pblh','rmol','akhs','akms'))
    f['u10e'][...] = cp.where(mask,f['u10'],f['u10e'])
    f['v10e'][...] = cp.where(mask,f['v10'],f['v10e'])
