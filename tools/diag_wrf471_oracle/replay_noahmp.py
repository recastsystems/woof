#!/usr/bin/env python3
"""Replay all 58 native amounts from unchanged real Noah-MP module entries."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from instrument_noahmp import INPUTS,AMOUNTS

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof import wrf_exact
 from woof.core.kernels import get_kernel,module_source,module_options
 assert wrf_exact.ENABLED
 run=json.loads(a.run_receipt.read_text());assert run['returncode']==0;root=a.run_receipt.resolve().parents[1];rows=[];captures=[]
 f=get_kernel('noahmp_output','native_noahmp_output')
 for row in run['captures']:
  path=root/row['path'];b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256']
  magic,nx,ny,step,opt,stepwtd,ni,na=map(int,np.frombuffer(b,'<i4',count=8));assert (magic,ni,na)==(47111,len(INPUTS),len(AMOUNTS));n=nx*ny
  now,dt,minutes=np.frombuffer(b,'<f4',count=3,offset=32);offset=44
  v=np.frombuffer(b,'<f4',count=ni*n,offset=offset).reshape(ni,ny,nx);offset+=ni*n*4
  snow=np.frombuffer(b,'<i4',count=n,offset=offset).reshape(ny,nx);offset+=n*4
  before=np.frombuffer(b,'<f4',count=na*n,offset=offset).reshape(na,ny,nx);offset+=na*n*4
  expected=np.frombuffer(b,'<f4',count=na*n,offset=offset).reshape(na,ny,nx);assert len(b)==offset+na*n*4
  out=[cp.asarray(x) for x in before];epoch=cp.zeros(2,dtype='f4')
  f(((n+127)//128,),(128,),(*(cp.asarray(x) for x in v),*out,cp.asarray(snow),epoch,np.float64(now),np.float32(dt),np.float32(minutes),np.int32(opt),np.int32(step),np.int32(stepwtd),np.int32(n)))
  for name,value,target in zip(AMOUNTS,out,expected):
   got=cp.asnumpy(value);diff=got.view('u4')!=target.view('u4')
   rows.append(dict(capture=row['path'],field=name,words=got.size,differing_words=int(diff.sum()),nonzero_words=int(np.count_nonzero(target)),max_abs=float(np.max(np.abs(target)))))
  captures.append(dict(capture=row['path'],itimestep=step,now_seconds=float(now),dt=float(dt),minutes=float(minutes),opt_run=opt,reset=bool(np.fmod(now,np.float32(60)*minutes)==0),nonzero_input_fields=[k for k,x in zip(INPUTS,v) if np.any(x)],snow_layers=snow.tolist()))
 words=sum(x['words'] for x in rows);diff=sum(x['differing_words'] for x in rows)
 a.output.write_text(json.dumps(dict(status='PASS' if not diff else 'FAIL',words=words,differing_words=diff,results=rows,captures=captures,run_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(module_source('noahmp_output').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('noahmp_output'))),undefined_exemptions=[]),indent=2)+'\n');print(words,'words',diff,'different');raise SystemExit(bool(diff))
if __name__=='__main__':main()
