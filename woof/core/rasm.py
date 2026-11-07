"""Public WRF RASM accumulator ownership and calendar alarms.

The clock decisions use WRF's REAL expression order. All field arithmetic
and counter updates execute in native_surface_diagnostics CUDA kernels.
"""
from __future__ import annotations
from datetime import timedelta
import math
import numpy as np
from woof.core.native_surface_diagnostics import SURFACE, MEANS, rasm_mean, rasm_diurnal
from woof.wrf_exact import ENABLED as WRF_EXACT

def _units(name):
    return ('Pa' if name in ('psfc','pmsl') else 'K' if name in ('tsk','t2','th2')
            else 'kg kg-1' if name=='q2' else 'm s-1' if name in ('u10','v10') else 'W m-2')

HISTORY = {f'rasm_{k}_mean': (k.upper()+'_M', _units(k), 'RASM interval mean '+k) for k in MEANS}
HISTORY.update({f'rasm_{k}_diurn': (k.upper()+'_D', _units(k), 'RASM monthly three-hour diurnal mean '+k) for k in SURFACE})
COUNTERS = ('rasm_nsteps_mean','rasm_nsteps_dtmp','rasm_nsteps_diurn')
TEMPORARY = tuple('rasm_'+k+'_dtmp' for k in SURFACE)
PERSISTENT = (*HISTORY,*TEMPORARY,*COUNTERS)
WORK = ('rasm_theta','rasm_p','rasm_pb')
NATIVE_SOURCES = {slot: row[0] for slot,row in HISTORY.items()}
NATIVE_SOURCES.update({slot:k.upper()+'_DTM' for slot,k in zip(TEMPORARY,SURFACE)})
NATIVE_SOURCES.update(zip(COUNTERS, ('NSTEP_M','NSTEP_DTM','NSTEP_D')))

def active(cfg): return bool(cfg.mean_diag or cfg.diurnal_diag)

def scratch_shapes(cfg):
    plane=(cfg.ny,cfg.nx)
    shapes={}
    if cfg.mean_diag:
        shapes.update({f'rasm_{k}_mean':plane for k in MEANS})
        shapes.update({k:plane for k in (*WORK,COUNTERS[0])})
    if cfg.diurnal_diag:
        shapes.update({f'rasm_{k}_diurn':(8,*plane) for k in SURFACE})
        shapes.update({k:plane for k in (*TEMPORARY,*COUNTERS[1:])})
    return shapes

def initialize(state,cfg):
    for name,shape in scratch_shapes(cfg).items():
        state.scratch(shape,name,dtype=np.int32 if name in COUNTERS else np.float32)

def _bound_clock(cfg):
    # Native diurnal rejects a rounded finish outside a three-hour clock
    # boundary. Keep that calendar check for fractional/non-dividing steps.
    return not WRF_EXACT and (not cfg.diurnal_diag or
                             (cfg.dt==int(cfg.dt) and 10800%cfg.dt==0))

def prepare(state,cfg):
    if cfg.mean_diag:
        if _bound_clock(cfg):
            # step() has already saved thp0. Nest feedback instead owns the
            # preceding solve's pre-feedback snapshot. Both stay resident.
            previous=getattr(state,'_wrf_th_phy_m_t0',None)
            if previous is None:previous=getattr(state,'thp0',None)
            if previous is not None:
                state._rasm_theta_bound=previous[0]
                return
            # Standalone callers need the same hold when no dycore owns it.
            state._rasm_theta_bound=state.existing_scratch('rasm_theta')
            state._rasm_theta_bound[...] = state.thp[0]
            return
        # th_phy_m_t0 is the preceding solve's final potential temperature.
        # It is held while the current solve advances t_2.
        previous=getattr(state,'_wrf_th_phy_m_t0',state.thp)
        state.existing_scratch('rasm_theta')[...] = previous[0]

def _nint(x): return math.floor(float(np.float32(x))+.5)

def alarm(seconds,dt,interval, *, monthly=False, now=None):
    """Native getResetState/getAvgState, including INT(dt) calendar steps."""
    if monthly:
        off=timedelta(seconds=int(dt))
        return now.month!=(now-off).month,now.month!=(now+off).month
    f=np.float32; x=f(f(seconds)/f(60)); d=f(dt)
    divisor=_nint(f(f(interval)/d))
    if divisor<=0: raise ValueError('RASM interval rounds to zero native steps')
    return (_nint(f(f(x*f(60))/d))%divisor==0,
            _nint(f(f(f(x+f(d/f(60)))*f(60))/d))%divisor==0)

def update(state,cfg):
    if not active(cfg): return
    if _bound_clock(cfg):return _update_bound(state,cfg)
    fields=state.physics.fields
    # These are the radiation driver's actual held boundary flux carriers.
    # A missing producer is a broken runtime binding, never a zero fallback.
    surface={k:fields[k] for k in SURFACE}
    start=state.physics.native_diagnostic_epoch
    now=start+timedelta(seconds=state.elapsed_seconds)
    s=state.existing_scratch
    if cfg.mean_diag:
        interval=cfg.mean_interval*{1:1,2:60,3:3600,4:86400,5:1}[cfg.mean_freq]
        reset,finish=alarm(state.elapsed_seconds,cfg.dt,interval,monthly=cfg.mean_freq==5,now=now)
        import cupy as cp
        pb=state.pb[0] if state.pb.ndim==3 else state.pb[0]
        s('rasm_pb')[...] = pb
        cp.subtract(state.p[0],s('rasm_pb'),out=s('rasm_p'))
        rasm_mean(surface,dict(t=s('rasm_theta'),p=s('rasm_p'),pb=s('rasm_pb'),moist=state.qv[0],ht=state.ht),
            {k:s('rasm_'+k+'_mean') for k in MEANS},s(COUNTERS[0]),reset=reset,finish=finish,_validated=True)
    if cfg.diurnal_diag:
        reset,finish=alarm(state.elapsed_seconds,cfg.dt,10800)
        rm,fm=alarm(state.elapsed_seconds,cfg.dt,1,monthly=True,now=now)
        hour=(now+timedelta(seconds=int(cfg.dt))).hour
        if finish and hour%3: raise RuntimeError('native RASM cannot select this diurnal cycle')
        cycle=hour//3 or 8
        rasm_diurnal(surface,{k:s('rasm_'+k+'_dtmp') for k in SURFACE},
            {k:s('rasm_'+k+'_diurn') for k in SURFACE},s(COUNTERS[1]),s(COUNTERS[2]),
            reset=reset,reset_month=rm,finish=finish,finish_month=fm,cycle=cycle,_validated=True)

def _bind_calls(state,cfg):
    """Bind run-owned surface and accumulator arrays once per configuration.

    Physics updates these carriers in place. A new driver or configuration
    rebuilds the binding; checkpoint readers rebuild it from their owners.
    Only the held theta view and clock scalars change between solves.
    """
    from woof.core.kernels import get_kernel
    n=cfg.nx*cfg.ny
    grid=((n+127)//128,);block=(128,)
    f=state.physics.fields;s=state.existing_scratch
    surface=tuple(f[k] for k in SURFACE)
    dt=np.float32(cfg.dt);zero=np.int32(0)
    mean=diurnal=None
    if cfg.mean_diag:
        interval=cfg.mean_interval*{1:1,2:60,3:3600,4:86400,5:1}[cfg.mean_freq]
        mean=(get_kernel('native_surface_diagnostics','rasm_mean_bound'),
            [*surface,s('rasm_theta'),state.p[0],state.pb[0],state.qv[0],state.ht,
             *(s('rasm_'+k+'_mean') for k in MEANS),s(COUNTERS[0]),np.int32(n),
             np.float32(0),dt,np.float32(interval),np.int32(cfg.mean_freq==5),
             zero,zero,np.int32(state.pb.ndim==3)])
    if cfg.diurnal_diag:
        diurnal=(get_kernel('native_surface_diagnostics','rasm_diurnal_bound'),
            [*surface,*(s('rasm_'+k+'_dtmp') for k in SURFACE),
             *(s('rasm_'+k+'_diurn') for k in SURFACE),s(COUNTERS[1]),s(COUNTERS[2]),
             np.int32(n),np.float32(0),dt,zero,zero,zero])
    calls=(cfg,f,grid,block,mean,diurnal,tuple(np.int32(k) for k in range(9)))
    state._rasm_bound_calls=calls
    return calls

def _update_bound(state,cfg):
    """Update clock words and launch the prebound resident arguments."""
    calls=getattr(state,'_rasm_bound_calls',None)
    if calls is None or calls[0] is not cfg or calls[1] is not state.physics.fields:
        calls=_bind_calls(state,cfg)
    _,_,grid,block,mean,diurnal,integers=calls
    seconds=np.float32(state.elapsed_seconds)
    rm=fm=cycle=integers[0]
    if cfg.diurnal_diag or cfg.mean_freq==5:
        now=state.physics.native_diagnostic_epoch+timedelta(seconds=state.elapsed_seconds)
        off=timedelta(seconds=int(cfg.dt));after=now+off
        rm=integers[now.month!=(now-off).month]
        fm=integers[now.month!=after.month]
        cycle=integers[after.hour//3 or 8]
    if mean is not None:
        fn,args=mean
        args[17]=getattr(state,'_rasm_theta_bound',args[17])
        args[-7]=seconds;args[-3]=rm;args[-2]=fm
        fn(grid,block,args)
    if diurnal is not None:
        fn,args=diurnal
        args[-5]=seconds;args[-3]=rm;args[-2]=fm;args[-1]=cycle
        fn(grid,block,args)

def registry_rows(registry):
    for key in ('mean_diag','diurnal_diag','mean_freq','mean_interval'):
        registry['parameters'][key]={'type':'integer','default':0,'minimum':0,
            'per_domain':key=='mean_interval','consuming_read':'woof/core/rasm.py'}
    registry.setdefault('diagnostic_packages',{}).update({key:{'1':{
        'native_source':'WRF v4.7.1 phys/module_diag_rasm.F','licence':'WRF public domain',
        'maturity':'supported','consumer':'woof/core/rasm.py:update',
        'oracle':'tools/surface_diagnostics_wrf461_oracle',
        'evidence_limit':'Real native reset, accumulation, interval finish and diurnal-bin finish replay; three-hour stock go/native comparison, native warm start, checkpoint, both feedback modes and ordinary ensemble. Monthly calendar has a separate native alarm check; no month-long forecast or observation skill claim.'}} for key in ('mean_diag','diurnal_diag')})
    registry.setdefault('diagnostic_state',{})['rasm']={native.lower():{'slot':slot,'restart':True,
        'history':slot in HISTORY,'type':'integer' if slot in COUNTERS else 'real'} for slot,native in NATIVE_SOURCES.items()}
