#!/usr/bin/env python3
"""Strict CUDA replay of real native AFWA input/result captures, no host physics."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
VOLUMES=('qv','qc','dz','rho','temperature','z','qr','qs','qg','u','v','pressure')
PLANES=('terrain','psfc','t2','q2','u10','v10','raincv','rainncv','rainc','rainnc','swdown')
PTYPE=('afwa_rain','afwa_snow','afwa_ice','afwa_fzra','afwa_snowfall')
OUTPUTS=('afwa_precip','afwa_totprecip','afwa_pwat','afwa_mslp','afwa_heatidx','afwa_wchill','afwa_fits',*PTYPE,'vil','radarvil','refd','refd_com','echotop','afwa_vis','afwa_vis_dust','afwa_vis_alpha')
def decode(path):
 raw=path.read_bytes();magic,nx,ny,nz,nphys,step,output,first=map(int,np.frombuffer(raw,dtype='<i4',count=8));assert magic==47105
 off=32;data={}
 for name,shape in [(n,(ny,nz,nx)) for n in VOLUMES]+[(n,(ny,nx)) for n in (*PLANES,*(n+'_before' for n in PTYPE),*OUTPUTS)]:
  count=int(np.prod(shape));a=np.frombuffer(raw,dtype='<f4',count=count,offset=off).reshape(shape)
  data[name]=np.ascontiguousarray(a.transpose(1,0,2) if len(shape)==3 else a);off+=count*4
 assert off==len(raw)
 return dict(nx=nx,ny=ny,nz=nz,physical_levels=nphys,step=step,output=bool(output),first=bool(first)),data

def replay(h,d):
 import cupy as cp
 from woof.core.kernels import get_kernel
 g={k:cp.asarray(v) for k,v in d.items()};n=h['nx']*h['ny'];nz=h['nz'];physical=h['physical_levels'];shape=(h['ny'],h['nx']);i=np.int32
 out={k:cp.zeros(shape,dtype='f4') for k in OUTPUTS}
 for k in PTYPE:out[k][...]=g[k+'_before']
 launch=(((n+127)//128,),(128,))
 get_kernel('afwa_diagnostics','afwa_base')(*launch,(g['qv'],g['rho'],g['dz'],g['temperature'][0],g['z'][0],g['terrain'],g['psfc'],*(g[k] for k in ('raincv','rainncv','rainc','rainnc')),*(out[k] for k in OUTPUTS[:4]),i(nz),i(n),i(1),i(h['output'])))
 get_kernel('afwa_diagnostics','afwa_ptype')(*launch,(*(g[k] for k in ('temperature','pressure','qv','z','dz','terrain','t2','swdown')),out['afwa_precip'],*(out[k] for k in PTYPE),np.float32(264.15),np.float32(50.),i(physical),i(n)))
 if h['output']:
  get_kernel('afwa_diagnostics','afwa_thermal')(*launch,(*(g[k] for k in ('t2','psfc','q2','u10','v10')),*(out[k] for k in OUTPUTS[4:7]),i(n)))
  ze=cp.zeros((nz,*shape),dtype='f4');cnt=n*nz
  get_kernel('afwa_diagnostics','afwa_dbz')(((cnt+127)//128,),(128,),(g['rho'],g['temperature'],g['qr'],g['qs'],g['qg'],ze,i(cnt)))
  get_kernel('afwa_diagnostics','afwa_radar_vil')(*launch,(*(g[k] for k in ('qr','qs','qg','rho','dz','z')),ze,*(out[k] for k in ('vil','radarvil','refd','refd_com','echotop')),i(physical),i(n),i(1),i(1)))
  get_kernel('afwa_diagnostics','afwa_visibility')(*launch,(*(g[k] for k in ('temperature','pressure','qv','z','u','v','terrain','qr','qs','qg','rho','psfc','t2','q2')),out['afwa_pwat'],*(out[k] for k in ('afwa_vis','afwa_vis_dust','afwa_vis_alpha')),i(physical),i(n)))
 return {k:cp.asnumpy(v) for k,v in out.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 from woof import wrf_exact
 from woof.core.kernels import module_source,module_options
 assert wrf_exact.ENABLED,'strict numerics required'
 native=json.loads(a.run_receipt.read_text());root=a.run_receipt.resolve().parents[1]
 if native['returncode'] != 0:
  assert native.get('completed') is True,'native run must complete'
  witness=native['completion_witness'];log=root/witness['path']
  assert hashlib.sha256(log.read_bytes()).hexdigest()==witness['sha256']
  assert witness['literal'] in log.read_text(),'native completion witness missing'
 results=[]
 for row in native['captures']:
  path=root/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  h,d=decode(path);actual=replay(h,d)
  for k in OUTPUTS if h['output'] else (*OUTPUTS[:3],*PTYPE):
   ref=d[k];v=actual[k];diff=ref.view('u4')!=v.view('u4');rec=dict(capture=row['path'],step=h['step'],field=k,words=ref.size,differing_words=int(diff.sum()))
   if diff.any():
    idx=tuple(map(int,np.argwhere(diff)[0]));rec['first_difference']=dict(index=idx,native=float(ref[idx]),cuda=float(v[idx]),native_word=hex(int(ref.view('u4')[idx])),cuda_word=hex(int(v.view('u4')[idx])))
   results.append(rec)
 words=sum(r['words'] for r in results);diff=sum(r['differing_words'] for r in results)
 rec=dict(status='PASS' if not diff else 'FAIL',words=words,differing_words=diff,results=results,native=native,native_receipt_sha256=hashlib.sha256(a.run_receipt.read_bytes()).hexdigest(),cuda_source_sha256=hashlib.sha256(module_source('afwa_diagnostics').encode()).hexdigest(),compiler_options=list(wrf_exact.effective_options(module_options('afwa_diagnostics'))),undefined_exemptions=[])
 a.output.write_text(json.dumps(rec,indent=2)+'\n');print(rec['status'],words,'words',diff,'different');raise SystemExit(bool(diff))
if __name__=='__main__':main()
