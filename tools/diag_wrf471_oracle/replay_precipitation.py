#!/usr/bin/env python3
"""Replay captured real native precipitation inputs on the production CUDA entry."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
NAMES=('raincv','rainncv','snowncv','t2','prec_acc_c','prec_acc_nc','snow_acc_nc')
def decode(path):
 b=path.read_bytes();magic,nx,ny,step=map(int,np.frombuffer(b,'<i4',count=4));assert magic==47106
 seconds,minutes=np.frombuffer(b,'<f4',count=2,offset=16);n=nx*ny
 values=np.frombuffer(b,'<f4',offset=24).reshape(10,ny,nx)
 return dict(nx=nx,ny=ny,step=step,seconds=float(seconds),minutes=float(minutes)),dict(zip((*NAMES,'pc_out','pn_out','sn_out'),values))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 from woof import wrf_exact
 from woof.core.kernels import get_kernel,module_source,module_options
 import cupy as cp
 assert wrf_exact.ENABLED
 run=json.loads(a.run_receipt.read_text());assert run['returncode']==0;root=a.run_receipt.resolve().parents[1];rows=[]
 for row in run['captures']:
  path=root/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  h,d=decode(path);g={k:cp.asarray(v.copy()) for k,v in d.items()};n=h['nx']*h['ny'];z=cp.zeros((h['ny'],h['nx']),dtype='f4');epoch=cp.zeros((2,),dtype='f4')
  get_kernel('output_diagnostics','native_precipitation_buckets')(((n+255)//256,),(256,),
    (g['raincv'],g['rainncv'],g['snowncv'],z,g['t2'],g['prec_acc_c'],g['prec_acc_nc'],g['snow_acc_nc'],z,epoch,np.int32(n),np.float64(h['seconds']),np.float32(h['minutes'])))
  for k,out in zip(NAMES[-3:],('pc_out','pn_out','sn_out')):
   actual=cp.asnumpy(g[k]);reference=d[out];diff=actual.view('u4')!=reference.view('u4')
   rows.append(dict(capture=row['path'],field=k,**h,words=n,differing_words=int(diff.sum()),nonzero_words=int(np.count_nonzero(reference)),maximum=float(np.max(reference))))
 words=sum(x['words'] for x in rows);diff=sum(x['differing_words'] for x in rows)
 a.output.write_text(json.dumps(dict(status='PASS' if diff==0 else 'FAIL',words=words,differing_words=diff,results=rows,run_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(module_source('output_diagnostics').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('output_diagnostics'))),undefined_exemptions=[]),indent=2)+'\n')
 print(words,'words',diff,'different');raise SystemExit(bool(diff))
if __name__=='__main__':main()
