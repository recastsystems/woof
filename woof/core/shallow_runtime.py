"""Bind native shallow-driver arguments and couple the returned tendencies."""
from __future__ import annotations
import cupy as cp
import numpy as np


def before_compose(driver, atmosphere, state, cfg):
    shcu = cfg.shcu_physics
    if shcu == 5:
        from woof.core.deng import step
        return step(driver, atmosphere, state, cfg)
    if shcu not in (3, 4):
        return
    from woof.core.cam5_runtime import workspace
    from woof.core.physics import couple_column_tendencies
    a = atmosphere
    f = driver.fields
    owner = workspace(driver)
    shape = state.p.shape[1:]
    zero = owner.array('shallow/zero_surface', shape, cp.float32, clear=True)
    raincv = driver.cu_raincv if cfg.cu_physics else zero
    if shcu == 3:
        from woof.core.grims import GrimsShallowCumulus
        scheme = getattr(driver, '_grims_shallow', None)
        if scheme is None:
            driver._grims_shallow = scheme = GrimsShallowCumulus(state.znu, state.p.shape[0])
        out = scheme(qv=a['qv'], temperature=a['temperature'],
                     pressure=a['pressure'], exner=a['exner'],
                     p_interface=a['p_interface'], z_interface=a['z_interface'],
                     wstar=f.get('shcu_wstar', zero), pblh=f['pblh'],
                     delta=f.get('shcu_delta', zero), raincv=raincv,
                     kpbl=f['kpbl'], dt=cfg.dt)
        driver.shallow_tendencies = couple_column_tendencies(
            state, cfg, rtheta=out.rthshten, rqv=out.rqvshten)
    else:
        from woof.core.shcu_nscv import launch_shcu_nscv
        held = getattr(driver, '_nscv_output', None)
        out = launch_shcu_nscv(
            t=a['temperature'], qv=a['qv'], qc=a['qc'], qi=a['qi'],
            p=a['pressure'], p8w=a['p_interface'], pi=a['exner'], rho=a['rho'],
            u=a['u'], v=a['v'], dz8w=a['dz'], w=state.w,
            raincv=raincv, xland=f['xland'], hpbl=f['pblh'],
            hfx=f['hfx'], qfx=f['qfx'], hbot=f['hbot'], htop=f['htop'],
            dt=cfg.dt, mp_physics=cfg.mp_physics, pgcon=cfg.sas_pgcon,
            f_qc=getattr(state, 'qc', None) is not None,
            f_qi=getattr(state, 'qi', None) is not None, out=held)
        driver._nscv_output = out
        driver.shallow_tendencies = couple_column_tendencies(
            state, cfg, ru=out['rushten'], rv=out['rvshten'],
            rtheta=out['rthshten'], rqv=out['rqvshten'],
            rqc=out['rqcshten'] if state.qc is not None else None,
            rqi=out['rqishten'] if getattr(state, 'qi', None) is not None else None)
        increment = out['pratesh'] * np.float32(cfg.dt)
        f['rainsh'] += increment
        driver._pending_rainbl += cp.maximum(increment, np.float32(0.))
        if driver.ruc_params is not None or driver.noahmp_params is not None:
            f['surface_raincv'] += cp.maximum(increment, np.float32(0.))
    driver.call_counts['shallow_cumulus'] = driver.call_counts.get('shallow_cumulus', 0) + 1
