#!/usr/bin/env python3
"""Strict production CUDA replay on real native AFWA parcel inputs."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from instrument_buoy import FIELDS

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',type=Path,required=True);p.add_argument('--capture-dir',type=Path);p.add_argument('--active-levels',type=int);p.add_argument('--active-depth-witness',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof import wrf_exact
 from woof.core.kernels import get_kernel,module_source,module_options
 assert wrf_exact.ENABLED
 run=json.loads(a.run_receipt.read_text());assert run['returncode']==0;root=a.run_receipt.resolve().parents[1];rows=[];inputs=[]
 captures=run.get('captures') or [dict(path=str(x.resolve().relative_to(root)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sorted(a.capture_dir.glob('*.bin'))]
 assert captures
 from woof.core.afwa_diagnostics import launch_buoyancy
 for row in captures:
  path=root/row['path'];b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256']
  magic,nx,ny,nz=map(int,np.frombuffer(b,'<i4',count=4));assert magic==47112;n=nx*ny
  v=np.frombuffer(b,'<f4',count=5*n*nz,offset=16).reshape(5,ny,nz,nx).transpose(0,2,1,3).copy();offset=16+5*n*nz*4
  ht=np.frombuffer(b,'<f4',count=n,offset=offset).reshape(ny,nx);offset+=n*4
  expected=np.frombuffer(b,'<f4',offset=offset).reshape(7,ny,nx)
  allocated=nz
  if a.active_levels is not None and a.active_levels!=allocated:
   assert a.active_depth_witness is not None
   w=np.frombuffer(a.active_depth_witness.read_bytes(),'<i4',count=4);assert w[0]==47112 and w[3]==a.active_levels
   nz=a.active_levels;assert 1<nz<allocated;v=v[:,:nz].copy()
  out=[cp.zeros((ny,nx),dtype='f4') for _ in FIELDS];work=[cp.zeros((2,nz,ny,nx),dtype='f4') for _ in range(3)]
  launch_buoyancy((*(cp.asarray(x) for x in v[:4]),cp.asarray(ht),*out,*work,cp.zeros((2,6,ny,nx),dtype='f4'),np.int32(nz),np.int32(n)))
  for name,value,target in zip(FIELDS,out,expected):
   got=cp.asnumpy(value);different=got.view('u4')!=target.view('u4')
   entries=[dict(index=list(map(int,ix)),actual=float(got[tuple(ix)]),expected=float(target[tuple(ix)]),actual_bits=hex(int(got.view('u4')[tuple(ix)])),expected_bits=hex(int(target.view('u4')[tuple(ix)]))) for ix in np.argwhere(different)[:8]]
   rows.append(dict(capture=row['path'],field=name,words=got.size,differing_words=int(different.sum()),nonzero_words=int(np.count_nonzero(target)),max_abs=float(np.max(np.abs(target))),examples=entries))
  inputs.append(dict(capture=row['path'],allocated_depth=allocated,active_depth=nz,top_values={k:dict(minimum=float(x[-1].min()),maximum=float(x[-1].max())) for k,x in zip(('t','p','qv','z','rh'),v)},surface_temperature_min=float(v[0,0].min()),surface_temperature_max=float(v[0,0].max())))
 words=sum(x['words'] for x in rows);diff=sum(x['differing_words'] for x in rows)
 a.output.write_text(json.dumps(dict(status='PASS' if not diff else 'FAIL',words=words,differing_words=diff,results=rows,inputs=inputs,run_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(module_source('afwa_buoyancy').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('afwa_buoyancy'))),undefined_exemptions=[],active_depth_witness=str(a.active_depth_witness),active_depth_witness_sha256=hashlib.sha256(a.active_depth_witness.read_bytes()).hexdigest() if a.active_depth_witness else None),indent=2)+'\n');print(words,'words',diff,'different');raise SystemExit(bool(diff))
if __name__=='__main__':main()
