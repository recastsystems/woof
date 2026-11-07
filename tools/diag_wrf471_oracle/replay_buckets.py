#!/usr/bin/env python3
"""Replay real strict native bucket captures through the production CUDA entry."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from instrument_buckets import NAMES

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof import wrf_exact
 from woof.core.kernels import get_kernel,module_source,module_options
 assert wrf_exact.ENABLED
 run=json.loads(a.run_receipt.read_text());assert run['returncode']==0;root=a.run_receipt.resolve().parents[1];rows=[]
 for row in run['captures']:
  path=root/row['path'];b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256'];magic,nx,ny,step=map(int,np.frombuffer(b,'<i4',count=4));assert magic==47107;n=nx*ny
  seconds,xtime,mm,joules=np.frombuffer(b,'<f4',count=4,offset=16);offset=32;groups=[]
  for dtype in ('<f4','<i4','<f4','<i4'):
   groups.append(np.frombuffer(b,dtype,count=18*n,offset=offset).reshape(18,ny,nx));offset+=18*n*4
  assert offset==len(b)
  amounts=[cp.asarray(x.copy()) for x in groups[0]];counters=[cp.asarray(x.copy()) for x in groups[1]];epoch=cp.zeros(2,dtype='f4')
  get_kernel('output_diagnostics','native_accumulation_buckets')(((n+255)//256,),(256,),(*amounts,*counters,epoch,np.int32(n),np.float64(seconds),xtime,mm,joules))
  for group,ref,names in ((amounts,groups[2],NAMES),(counters,groups[3],tuple('i_'+x for x in NAMES))):
   for actual,expected,name in zip(group,ref,names):
    value=cp.asnumpy(actual);diff=value.view('u4')!=expected.view('u4')
    rows.append(dict(capture=row['path'],field=name,step=step,seconds=float(seconds),words=n,differing_words=int(diff.sum()),nonzero_words=int(np.count_nonzero(expected)),maximum=float(expected.max())))
 words=sum(x['words'] for x in rows);diff=sum(x['differing_words'] for x in rows)
 a.output.write_text(json.dumps(dict(status='PASS' if diff==0 else 'FAIL',words=words,differing_words=diff,results=rows,run_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(module_source('output_diagnostics').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('output_diagnostics'))),undefined_exemptions=[]),indent=2)+'\n');print(words,'words',diff,'different');raise SystemExit(bool(diff))
if __name__=='__main__':main()
