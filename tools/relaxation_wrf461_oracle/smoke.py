"""Run a small dry engine integration and retain only full-field hashes."""
import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np


def run(selector,steps):
    import cupy as cp
    from woof.config import RunConfig,validate_run_config
    from woof.core.grid import make_base_state,make_vertical_coord
    from woof.core.state import init_at_rest
    from woof.core.dycore import step
    from woof.core.physics import initialize_physics
    from woof.wrf_exact import ENABLED
    cfg=RunConfig(nx=12,ny=8,nz=12,dx=10000.,dy=10000.,ztop=16000.,dt=10.,run_seconds=steps*10.,
                  ra_physics=0,ra_lw_physics=selector,ra_sw_physics=0,km_opt=1,moist=False,
                  radt_minutes=0.0,radt=0.0)
    validate_run_config(cfg)
    coord=make_vertical_coord(cfg.nz)
    base=make_base_state(coord,lambda z:np.full_like(z,300.),p_surf=cfg.p_surf,ztop=cfg.ztop)
    s=init_at_rest(cfg,coord,base)
    s.u[...] = 10.; s.v[...] = 3.
    if selector:
        latitude=cp.broadcast_to(cp.linspace(-80.,80.,cfg.ny,dtype=cp.float32)[:,None],(cfg.ny,cfg.nx)).copy()
        initialize_physics(s,cfg,radiation_start_time=datetime(2000,1,1),
                           radiation_latitude=latitude,radiation_longitude=cp.zeros_like(latitude))
    frames=[]
    for n in range(steps):
        step(s,cfg)
        hashes={}
        for name in ('u','v','w','thp','php','mup','p','al','alt'):
            a=cp.asnumpy(getattr(s,name)); assert np.isfinite(a).all(),name
            hashes[name]=hashlib.sha256(a.tobytes()).hexdigest()
        frames.append(hashes)
    return {'selector':selector,'strict':ENABLED,'steps':steps,'frames':frames,
            'radiation_calls':s.physics.radiation_callable.update_count if selector else 0,
            'temperature_span':float(cp.max(s.total_theta())-cp.min(s.total_theta()))}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--selector',type=int,default=0);p.add_argument('--steps',type=int,default=3);a=p.parse_args()
    r=run(a.selector,a.steps);Path(a.out).write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2))
