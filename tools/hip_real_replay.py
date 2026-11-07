#!/usr/bin/env python3
"""Replay committed real WRF inputs through production GPU launchers.

Run once per backend in separate processes; compare archives with --compare.
This is a real-field operator replay, not a whole-forecast qualification.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--compare',nargs=2,type=Path);a=p.parse_args()
 if a.compare:
  left,right=(np.load(x,allow_pickle=False) for x in a.compare)
  assert left.files==right.files
  rows={}
  for k in left.files:
   x,y=left[k],right[k];assert x.dtype==y.dtype and x.shape==y.shape
   rows[k]=dict(words=int(x.size),different_words=int(np.count_nonzero(x.view('u4')!=y.view('u4'))),cuda_sha256=hashlib.sha256(x.tobytes()).hexdigest(),hip_sha256=hashlib.sha256(y.tobytes()).hexdigest())
  report=dict(exact=all(r['different_words']==0 for r in rows.values()),fields=rows)
  print(json.dumps(report,indent=2));return 0 if report['exact'] else 1
 import woof,cupy as cp
 from woof.hip_backend import install,device_info
 install()
 from woof.verify.bigstep_prep_oracle import load_prep_fixture,prep_port_outputs
 out={}
 for case in ('real-periodic2','real-specified2','real-periodic5','real-specified5'):
  data=load_prep_fixture(case)
  for supplied in (False,True):
   for name,value in prep_port_outputs(data,supplied_face_masses=supplied).items():out[f'{case}/{int(supplied)}/{name}']=np.ascontiguousarray(value)
 cp.cuda.get_current_stream().synchronize()
 np.savez(a.output,**out)
 print(json.dumps(dict(device=device_info(),fields=len(out),words=sum(x.size for x in out.values())),indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())
