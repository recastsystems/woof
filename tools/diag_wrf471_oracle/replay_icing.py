#!/usr/bin/env python3
"""Strict replay of actual native AFWA icing module entries."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from instrument_icing import PLANES,RAW

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof import wrf_exact
 from woof.core.kernels import get_kernel,module_source,module_options
 assert wrf_exact.ENABLED
 run=json.loads(a.run_receipt.read_text());assert run['returncode']==0;root=a.run_receipt.resolve().parents[1];rows=[];inputs=[]
 from woof.core.afwa_diagnostics import launch_icing as f
 for row in run['captures']:
  path=root/row['path'];b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256']
  magic,nx,ny,nz,levels,opt,call,kpe=map(int,np.frombuffer(b,'<i4',count=8));assert magic==47110;n=nx*ny
  v=np.frombuffer(b,'<f4',count=7*n*nz,offset=32).reshape(7,ny,nz,nx).transpose(0,2,1,3).copy()
  planes=np.frombuffer(b,'<f4',count=7*n,offset=32+7*n*nz*4).reshape(7,ny,nx)
  raw=np.frombuffer(b,'<f4',offset=32+(7*n*nz+7*n)*4).reshape(2,ny,nz,nx).transpose(0,2,1,3).copy()
  out=[cp.zeros((ny,nx),dtype='f4') for _ in PLANES]+[cp.zeros((nz,ny,nx),dtype='f4') for _ in RAW]
  f((*(cp.asarray(x) for x in v),*out,np.int32(nz),np.int32(levels),np.int32(kpe-1),np.int32(opt),np.int32(n)))
  for name,value,expected in zip((*PLANES,*RAW),out,(*planes,*raw)):
   got=cp.asnumpy(value);diff=got.view('u4')!=expected.view('u4')
   rows.append(dict(capture=row['path'],field=name,call=call,icing_option=opt,words=got.size,differing_words=int(diff.sum()),nonzero_words=int(np.count_nonzero(expected)),max_abs=float(np.max(np.abs(expected)))))
  inputs.append(dict(capture=row['path'],allocated_depth=nz,active_levels=levels,native_kpe=kpe,cloud_number_min=float(v[6].min()),cloud_number_max=float(v[6].max()),supercooled_cloud_nonzero=int(np.count_nonzero((v[0]<273.15)&(v[5]>1.e-12))),supercooled_cloud_max=float(np.max(np.where(v[0]<273.15,v[5],0.)))))
 words=sum(x['words'] for x in rows);diff=sum(x['differing_words'] for x in rows)
 a.output.write_text(json.dumps(dict(status='PASS' if not diff else 'FAIL',words=words,differing_words=diff,results=rows,inputs=inputs,run_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(module_source('afwa_diagnostics').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('afwa_diagnostics'))),undefined_exemptions=[]),indent=2)+'\n');print(words,'words',diff,'different');raise SystemExit(bool(diff))
if __name__=='__main__':main()
