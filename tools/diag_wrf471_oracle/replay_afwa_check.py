#!/usr/bin/env python3
"""Strict replay of real AFWA held U/V/theta captures and native verdicts."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof import wrf_exact
 from woof.core.kernels import get_kernel,module_source,module_options
 assert wrf_exact.ENABLED
 receipt=json.loads(a.run_receipt.read_text());assert receipt['returncode']==0
 root=a.run_receipt.resolve().parents[1];rows=[]
 for row in receipt['captures']:
  path=root/row['path'];b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256']
  magic,nx,ny,nz,step,due,first,enabled=map(int,np.frombuffer(b,'<i4',count=8));assert magic==47108;n=nx*ny*nz
  inp=np.frombuffer(b,'<f4',count=3*n,offset=32).reshape(3,ny,nz,nx).transpose(0,2,1,3)
  expected=np.frombuffer(b,'<u4',count=1,offset=32+3*n*4);assert len(b)==36+3*n*4
  status=cp.zeros(1,dtype='u4')
  if enabled>0 and due and not first:
   arrays=[cp.asarray(x.copy()) for x in inp]
   get_kernel('afwa_diagnostics','afwa_bad_data_check')(((n+255)//256,),(256,),(*arrays,status,np.int32(n)))
  actual=cp.asnumpy(status);different=int(np.count_nonzero(actual!=expected))
  rows.append(dict(capture=row['path'],step=step,output_due=bool(due),native_first=bool(first),input_words=3*n,output_words=1,differing_words=different,expected=int(expected[0]),maximum_wind=float(np.nanmax(np.abs(inp[:2]))),theta_min=float(np.nanmin(inp[2])),theta_max=float(np.nanmax(inp[2]))))
 diff=sum(x['differing_words'] for x in rows)
 result=dict(status='PASS' if diff==0 else 'FAIL',words=len(rows),input_words=sum(x['input_words'] for x in rows),differing_words=diff,results=rows,undefined_exemptions=[],source_sha256=hashlib.sha256(module_source('afwa_diagnostics').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('afwa_diagnostics'))),run_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest())
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(len(rows),'native verdict words,',diff,'different; inputs',result['input_words']);raise SystemExit(bool(diff))
if __name__=='__main__':main()
