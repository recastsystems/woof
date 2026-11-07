#!/usr/bin/env python3
"""Complete moist RK3/acoustic/microphysics execution, separate-process parity."""
from pathlib import Path
import argparse,json
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import woof,cupy as cp
 from woof.hip_backend import install,device_info
 install()
 from woof.config import RunConfig,validate_run_config
 from woof.core.grid import make_base_state,make_vertical_coord
 from woof.core.moist import init_moist_balanced
 from woof.core.dycore import step
 from woof.core.physics import initialize_physics
 cfg=validate_run_config(RunConfig(nx=12,ny=10,nz=24,dx=3000.,dy=3000.,ztop=18000.,
     dt=2.,run_seconds=4.,moist=True,mp_physics=10,ra_lw_physics=0,ra_sw_physics=0))
 coord=make_vertical_coord(cfg.nz)
 base=make_base_state(coord,lambda z:300.+.003*np.asarray(z),p_surf=cfg.p_surf,ztop=cfg.ztop)
 state=init_moist_balanced(cfg,coord,base,lambda z:.01*np.exp(-np.asarray(z)/2200.))
 initialize_physics(state,cfg,landmask=1.,tsk=290.)
 for _ in range(2):step(state,cfg)
 cp.cuda.get_current_stream().synchronize()
 fields={k:cp.asnumpy(v) for k,v in vars(state).items() if isinstance(v,cp.ndarray)}
 selected=('u','v','w','thp','php','mup','qv','qc','qr','qi','qs','qg')
 for name in selected:assert np.isfinite(fields[name]).all(),name
 np.savez(a.output,**fields)
 print(json.dumps(dict(device=device_info(),steps=2,acoustic=True,mp_physics=10,
     fields=len(fields),words=sum(v.size for v in fields.values()),config=repr(cfg)),indent=2))
if __name__=='__main__':main()
