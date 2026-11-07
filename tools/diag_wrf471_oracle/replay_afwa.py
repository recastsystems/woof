#!/usr/bin/env python3
"""Replay bounded real AFWA base/thermal captures against native results.

No reference arithmetic is transcribed in Python. Native output words are
captured after the unchanged pinned WRF driver. The final memory row is
included in PWAT, as native Pwat uses kme (and native dz there is zero).
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
VOLUMES=('qv','qc','dz','rho','temperature','z')
PLANES=('terrain','psfc','t2','q2','u10','v10','raincv','rainncv','rainc','rainnc')
OUTPUTS=('afwa_precip','afwa_totprecip','afwa_pwat','afwa_mslp','afwa_heatidx','afwa_wchill','afwa_fits')
def decode(path):
 raw=path.read_bytes();header=np.frombuffer(raw,dtype='<i4',count=8);magic,nx,ny,nz,nphys,step,output,first=map(int,header)
 if magic!=47104:raise ValueError('unexpected native AFWA capture version')
 offset=32;data={}
 for name,shape in [(n,(ny,nz,nx)) for n in VOLUMES]+[(n,(ny,nx)) for n in (*PLANES,*OUTPUTS)]:
  count=int(np.prod(shape));a=np.frombuffer(raw,dtype='<f4',count=count,offset=offset).reshape(shape)
  data[name]=np.ascontiguousarray(a.transpose(1,0,2) if len(shape)==3 else a);offset+=count*4
 assert offset==len(raw),(offset,len(raw))
 return dict(nx=nx,ny=ny,nz=nz,physical_levels=nphys,step=step,output=bool(output),first=bool(first)),data

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import cupy as cp
 from woof import wrf_exact
 from woof.core.kernels import get_kernel,module_source,module_options
 if not wrf_exact.ENABLED:raise ValueError('requires strict native numerics before process startup')
 native=json.loads(a.run_receipt.read_text());assert native['returncode']==0,'native run must complete'
 root=a.run_receipt.resolve().parents[1];results=[]
 for row in native['captures']:
  path=root/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  h,d=decode(path);g={k:cp.asarray(v) for k,v in d.items()};n=h['nx']*h['ny'];nz=h['nz'];shape=(h['ny'],h['nx'])
  out={k:cp.zeros(shape,dtype='f4') for k in OUTPUTS}
  get_kernel('afwa_diagnostics','afwa_base')(((n+127)//128,),(128,),
      (g['qv'],g['rho'],g['dz'],g['temperature'][0],g['z'][0],g['terrain'],g['psfc'],
       *(g[k] for k in ('raincv','rainncv','rainc','rainnc')),*(out[k] for k in OUTPUTS[:4]),
       np.int32(nz),np.int32(n),np.int32(1),np.int32(h['output'])))
  if h['output']:
   get_kernel('afwa_diagnostics','afwa_thermal')(((n+127)//128,),(128,),
       (*(g[k] for k in ('t2','psfc','q2','u10','v10')),*(out[k] for k in OUTPUTS[4:]),np.int32(n)))
  for k in OUTPUTS if h['output'] else OUTPUTS[:3]:
   actual=cp.asnumpy(out[k]);ref=d[k];diff=actual.view('u4')!=ref.view('u4');r=dict(capture=row['path'],step=h['step'],output=h['output'],field=k,words=ref.size,differing_words=int(diff.sum()))
   if diff.any():
    idx=tuple(map(int,np.argwhere(diff)[0]));r['first_difference']=dict(index=idx,native=float(ref[idx]),cuda=float(actual[idx]),native_word=hex(int(ref.view('u4')[idx])),cuda_word=hex(int(actual.view('u4')[idx])))
   results.append(r)
 total=sum(x['words'] for x in results);diff=sum(x['differing_words'] for x in results)
 rec=dict(status='PASS' if diff==0 else 'FAIL',words=total,differing_words=diff,results=results,native=native,native_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest(),cuda_source_sha256=hashlib.sha256(module_source('afwa_diagnostics').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('afwa_diagnostics'))),device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),no_undefined_exemption=True)
 a.output.write_text(json.dumps(rec,indent=2)+'\n');print(rec['status'],total,'words',diff,'different');raise SystemExit(bool(diff))
if __name__=='__main__':main()
