"""Whole-domain MAD-WRF hooks; empty MP96 has no prep/finish arithmetic."""
from __future__ import annotations
from woof.core.madwrf_contract import TRACER_PAIRS

def initialize(state, cfg):
    if not (cfg.madwrf_opt == 2 or cfg.madwrf_cldinit):
        return {}
    import cupy as cp
    import numpy as np
    if cfg.madwrf_cldinit and not getattr(state, "_madwrf_native_clouds", False):
        from woof.core.madwrf import init_madwrf_clouds_device
        base = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        theta = state.thp + base
        if cfg.use_theta_m:
            theta = theta / (np.float32(1.) + np.float32(461.6 / 287.) * state.qv)
        missing = cp.full(state.p.shape[1:], np.float32(-999.9), cp.float32)
        obs = getattr(state, "_madwrf_observations", {})
        cf = init_madwrf_clouds_device(
            t=cp.ascontiguousarray(theta-np.float32(300.)),
            p=getattr(state,"_madwrf_hydrostatic_pressure",state.p),
            ph_2=state.php,
            phb=cp.ascontiguousarray(cp.broadcast_to(state.phb if state.phb.ndim==3 else state.phb[:,None,None],state.php.shape)),
            qv=state.qv,qc=state.qc,qi=state.qi,qs=state.qs,
            xland=state.physics.fields["xland"],ht=state.ht,
            **{name:obs.get(name,missing.copy()) for name in ("cldmask","cldtopz","cldbasez","brtemp")},
            **{"flag_"+name:int(name in obs) for name in ("cldmask","cldtopz","cldbasez","brtemp")},
            dx=cfg.dx,dy=cfg.dy,p00=100000.)
        if state.physics.cldfra is not None:
            state.physics.cldfra[...] = cf[:-1]
    if cfg.madwrf_opt == 2 and not getattr(state, "_madwrf_native_tracers", False):
        for water,tracer in TRACER_PAIRS:
            getattr(state,tracer)[...] = getattr(state,water)
    return {"madwrf": {"cloud_initialization":bool(cfg.madwrf_cldinit),
                       "tracers":cfg.madwrf_opt==2}}

def nudge(state,cfg):
    if cfg.madwrf_opt != 2:
        return False
    from woof.core.madwrf import cloud_tracer_nudge_device,minutes_since_start
    return cloud_tracer_nudge_device(
        *(getattr(state,name) for name in ("qc","qi","qs","tr_qc","tr_qi","tr_qs")),
        cfg.dt,cfg.madwrf_dt_relax,cfg.madwrf_dt_nudge,
        minutes_since_start(state.elapsed_seconds),
        spec_zone=cfg.spec_zone if cfg.specified or cfg.nested else 0,
        periodic_x=not (cfg.open_x or cfg.specified or cfg.nested))
