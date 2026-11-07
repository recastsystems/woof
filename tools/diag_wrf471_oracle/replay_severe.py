from pathlib import Path
import os,json,hashlib,sys
import numpy as np,cupy as cp
from woof.core.kernels import get_kernel
from woof import wrf_exact
assert wrf_exact.ENABLED, "Native replay requires GPUWM_WRF_EXACT=1"
from woof.core.afwa_severe import surface_buoyancy
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--captures',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fused',action='store_true');cli=parser.parse_args()
r=cli.captures.resolve().parent.parent;spec=json.loads(Path(__file__).with_name('severe-print-layout.json').read_text());rows=[]
previous={}
paths=sorted(cli.captures.resolve().glob('severe-*.bin'));assert paths, 'No real native severe captures'
for path in paths:
 raw=path.read_bytes();header=np.frombuffer(raw,dtype='<i4',count=6);magic,nx,ny,nz,step,buoy=map(int,header);assert magic==47115;dt=np.frombuffer(raw,dtype='<f4',count=1,offset=24)[0];offset=28;n=nx*ny
 def read(shape):
  global offset
  count=int(np.prod(shape));a=np.frombuffer(raw,dtype='<f4',count=count,offset=offset).copy().reshape(shape);offset+=count*4;return a
 h={k:read((ny,nz,nx)).transpose(1,0,2).copy() for k in spec['severe_volumes']};h.update({k:read((ny,nx)) for k in spec['severe_planes']});expected={k:read((ny,nx)) for k in spec['severe_outputs']};assert offset==len(raw)
 d={k:cp.asarray(v) for k,v in h.items()};i32=np.int32;launch=(((n+127)//128,),(128,))
 extra={};extra_words=0
 def compare_extra(k,a,b):
  global extra_words
  a=np.ascontiguousarray(a);b=np.ascontiguousarray(b);extra_words+=a.size
  bad=a.view('u4')!=b.view('u4')
  if bad.any():extra[k]=dict(words=int(bad.sum()),max_abs=float(np.nanmax(np.abs(a-b))))
 rh=cp.empty_like(d['rh'])
 get_kernel('afwa_severe','afwa_severe_humidity')(((n*nz+255)//256,),(256,),(d['p'],d['t_phy'],d['qv'],rh,i32(n*nz)))
 compare_extra('relative_humidity',cp.asnumpy(rh)[:,1:-1,1:-1],h['rh'][:,1:-1,1:-1])
 # Instantaneous native shear is independent of prior interval maxima.
 # The first two captures also supply the actual preceding gust state.
 pp=path.with_name(path.name.replace('severe-','print-'))
 if pp.exists():
  rawp=pp.read_bytes();_,px,py,_,_=np.frombuffer(rawp,dtype='<i4',count=5)
  rain=np.frombuffer(rawp,dtype='<f4',count=11*px*py,offset=28).reshape(11,py,px)
  precip=cp.asarray((rain[4,2:2+ny,2:2+nx]+rain[5,2:2+ny,2:2+nx]).copy())
 else:precip=cp.zeros((ny,nx),dtype='f4')
 key=path.name.rsplit('-',1)[0]
 prior=cp.asarray(previous.get(key,np.zeros((ny,nx),dtype='f4')))
 work=[cp.zeros((ny,nx),dtype='f4') for _ in range(11)]
 get_kernel('afwa_severe','afwa_severe_driver')(*launch,
  (d['u_phy'],d['v_phy'],d['z'],d['ht'],d['u10'],d['v10'],precip,prior,work[0],work[1],*work[2:],i32(step==1),dt,i32(nz),i32(n)))
 compare_extra('instantaneous_llws',cp.asnumpy(work[1])[1:-1,1:-1],h['llws'][1:-1,1:-1])
 if step<=2 and pp.exists():compare_extra('interval_gust',cp.asnumpy(prior)[1:-1,1:-1],h['wspd10max'][1:-1,1:-1])
 previous[key]=h['wspd10max']

 get_kernel('afwa_severe','afwa_severe_column')(*launch,(d['w_2'],d['t_phy'],d['z'],d['p'],d['qi'],d['qs'],d['qg'],d['rho'],d['dz8w'],d['rh'],d['ht'],d['t2'],d['uh'],d['midrh_min_old'],d['midrh_min'],d['w_up_max'],d['w_dn_max'],d['afwa_hail'],d['tcoli_max'],d['grpl_flx_max'],i32(nz),i32(n),d['qv'],i32(cli.fused)))
 if buoy:
  work=[cp.zeros((2,nz,ny,nx),dtype='f4') for _ in range(3)];meta=cp.zeros((2,6,ny,nx),dtype='f4');dummy=[cp.zeros((ny,nx),dtype='f4') for _ in range(2)]
  surface_buoyancy((d['t_phy'],d['p'],d['qv'],d['z'],d['ht'],*(d[k] for k in ('cape','cin','zlfc','plfc','lidx')),*dummy,*work,meta,i32(nz),i32(n)))
 prev=d['tornado_mask'].copy();dur=d['tornado_dur'].copy()
 get_kernel('afwa_severe','afwa_severe_tornado')(*launch,(d['uh'],d['llws'],d['zlfc'],d['midrh_min'],d['midrh_min_old'],prev,dur,d['tornado_mask'],d['tornado_dur'],d['afwa_tornado'],dt,i32(ny),i32(nx),i32(1),i32(1),i32(1),i32(1),i32(0),i32(0)))
 diffs=dict(extra);words=extra_words
 for k,v in expected.items():
  a=cp.asnumpy(d[k])[1:-1,1:-1];b=v[1:-1,1:-1];words+=a.size;diff=int(np.count_nonzero(a.view('u4')!=b.view('u4')))
  if diff:diffs[k]=dict(words=diff,max_abs=float(np.nanmax(np.abs(a-b))),first=[float(x) for x in (a[(a.view('u4')!=b.view('u4'))][0],b[(a.view('u4')!=b.view('u4'))][0])])
 row=dict(path=str(path.relative_to(r)),sha256=hashlib.sha256(raw).hexdigest(),step=step,buoy=buoy,words=words,diffs=diffs);rows.append(row);print(row,flush=True)
result=dict(status='PASS' if all(not x['diffs'] for x in rows) else 'FAIL',words=sum(x['words'] for x in rows),captures=rows)
cli.output.write_text(json.dumps(result,indent=2)+'\n');assert result['status']=='PASS'
