#!/usr/bin/env python3
"""Run whole native-init MPAS dry timesteps for CUDA/HIP comparison."""
from pathlib import Path
import argparse,json,sys
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--grid',type=Path,required=True);p.add_argument('--init',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import woof,cupy as cp
 from woof.hip_backend import install,device_info
 install()
 from check_samehost_dycore_v841 import prepare,_driver,step_and_publish
 from woof.hex.config_v841 import V841DryDycoreConfig
 from woof.hex.cuda_backend.runtime import KernelCache,require_cuda
 from woof.hex.cuda_dualrun import STATE_FIELDS,SIDECAR_FIELDS
 from dataclasses import replace
 from run_cuda_v841_jw_dualrun import _config
 cfg=replace(_config(2.,6),config_smdiv=V841DryDycoreConfig().config_smdiv)
 prepared=prepare(a.grid,a.init,cfg)
 cache=KernelCache(capability=require_cuda())
 driver=_driver(prepared,cfg,cache)
 for _ in range(2):step_and_publish(driver)
 cp.cuda.get_current_stream().synchronize()
 out={}
 for group,names in ((driver.atmosphere.state,STATE_FIELDS),(driver.atmosphere.saved,SIDECAR_FIELDS)):
  for name in names:
   value=getattr(group,name)
   if isinstance(value,cp.ndarray):out[name]=cp.asnumpy(value)
 for name,value in out.items():assert np.isfinite(value).all(),name
 np.savez(a.output,**out)
 receipt=dict(device=device_info(),steps=2,fields=len(out),words=sum(v.size for v in out.values()),
     grid=str(a.grid),init=str(a.init),compile_manifest=cache.compile_manifest())
 a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='compile_manifest'}))
if __name__=='__main__':main()
