#!/usr/bin/env python3
"""Replay real native sixteen-rate accumulation, unchanged module boundary."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from instrument_tendencies import ACC,RATES

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof import wrf_exact
 from woof.core.kernels import get_kernel,module_source,module_options
 assert wrf_exact.ENABLED
 run=json.loads(a.run_receipt.read_text());assert run['returncode']==0;root=a.run_receipt.resolve().parents[1];rows=[]
 f=get_kernel('output_diagnostics','native_physics_tendency_accumulation')
 for row in run['captures']:
  path=root/row['path'];b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256']
  magic,nx,ny,nz,step,cu,sh,minutes=map(int,np.frombuffer(b,'<i4',count=8));assert magic==47109;n=nx*ny*nz
  seconds,dt=np.frombuffer(b,'<f4',count=2,offset=32)
  arrays=np.frombuffer(b,'<f4',offset=40).reshape(48,ny,nz,nx).transpose(0,2,1,3).copy()
  assert len(b)==40+48*n*4
  rates=[cp.asarray(v) for v in arrays[:16]];amounts=[cp.asarray(v) for v in arrays[16:32]];epoch=cp.zeros(2,dtype='f4');null=np.uint64(0)
  f(((n+255)//256,),(256,),(*rates,*amounts,null,null,epoch,np.int32(n),np.int32(nx),np.int32(ny),np.int32(0),np.int32(1),np.int32(cu),np.int32(sh),np.float64(seconds),np.float32(minutes*60.),dt))
  for name,value,expected in zip(ACC,amounts,arrays[32:]):
   got=cp.asnumpy(value);diff=got.view('u4')!=expected.view('u4')
   rows.append(dict(capture=row['path'],field=name,step=step,seconds=float(seconds),cu_used=cu,shcu_used=sh,words=n,differing_words=int(diff.sum()),nonzero_words=int(np.count_nonzero(expected)),max_abs=float(np.max(np.abs(expected)))))
 words=sum(x['words'] for x in rows);diff=sum(x['differing_words'] for x in rows)
 a.output.write_text(json.dumps(dict(status='PASS' if diff==0 else 'FAIL',words=words,differing_words=diff,results=rows,run_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(module_source('output_diagnostics').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('output_diagnostics'))),undefined_exemptions=[],input_contract='Actual module rates after completed-mass uncoupling; native inactive CU/SH inputs are never consumed, matching native IF guards'),indent=2)+'\n');print(words,'words',diff,'different');raise SystemExit(bool(diff))
if __name__=='__main__':main()
