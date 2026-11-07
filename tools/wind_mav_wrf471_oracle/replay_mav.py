from pathlib import Path
import json,hashlib,time,os
import numpy as np
import cupy as cp
from woof.wrf_exact import ENABLED, compile_receipt
from woof.core.wind_mav import MavWorkspace
import argparse
ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path);ap.add_argument('--debug',type=Path);ap.add_argument('--allow-differences',action='store_true',help='diagnostic fast-mode profile only');args=ap.parse_args()
p=args.capture;raw=(p/'mav-native.in').read_bytes();at=0
def read(n,kind='f4'):
 global at
 a=np.frombuffer(raw,dtype='>'+kind,count=n,offset=at).astype(kind);at+=n*4;return a
nx,ny,nzw,nt,model,overlap=map(int,read(6,'i4'));nz=nzw;nc=nx*ny
meta=read(3);diam,hub,area=[read(nt) for _ in range(3)];ix,jy=[read(nt,'i4') for _ in range(2)];xt,yt=[read(nt) for _ in range(2)];nv=read(nt,'i4');stc,stc2=[read(nt) for _ in range(2)]
ws,tc,pc=[read(nt*100).reshape(100,nt) for _ in range(3)]
def vol():return read(nx*nzw*ny).reshape(ny,nzw,nx).transpose(1,0,2).copy()
u,v,dz=[vol() for _ in range(3)];zw=read(nx*(nzw+1)*ny).reshape(ny,nzw+1,nx).transpose(1,0,2).copy();zm=vol();land,co,si=[read(nc).reshape(ny,nx) for _ in range(3)];du,dv=[vol() for _ in range(2)]
assert at==len(raw)
out=(p/'mav-native.out').read_bytes();r=np.frombuffer(out,dtype='>f4').astype('f4');expected={}
for i,name in enumerate(['du','dv','tke']):expected[name]=r[i*nzw*nc:(i+1)*nzw*nc].reshape(ny,nzw,nx).transpose(1,0,2)[:nz].copy()
expected['POWER']=r[3*nzw*nc:].reshape(ny,nx)
gpu=lambda a:cp.asarray(np.ascontiguousarray(a))
w=MavWorkspace((nz,ny,nx),gpu((jy-1)*nx+ix-1),gpu(xt),gpu(yt),gpu(hub),gpu(diam),gpu(area),gpu(nv),gpu(ws),gpu(tc),gpu(pc),gpu(stc),gpu(stc2),dx=meta[0],dt=meta[1],correction=meta[2],model=model,overlap=overlap)
a={k:gpu(vv) for k,vv in dict(u=u[:nz],v=v[:nz],dz=dz[:nz],z_interface=zw).items()};ddu=gpu(du[:nz]);ddv=gpu(dv[:nz]);q=cp.zeros((nz,ny,nx),cp.float32)
w.apply(a,gpu(land),gpu(co),gpu(si),ddu,ddv,q,q.copy(),z_mass=gpu(zm[:nz]));cp.cuda.get_current_stream().synchronize()
actual={'du':cp.asnumpy(ddu),'dv':cp.asnumpy(ddv),'tke':cp.asnumpy(w.tke_increment),'POWER':cp.asnumpy(w.power)}
res={}
for k,b in expected.items():
 a=actual[k];neq=a.view('u4')!=b.view('u4');indices=np.argwhere(neq);res[k]={'words':a.size,'different':int(neq.sum()),'max_absolute':float(np.max(abs(a-b))),'first':[{'index':ix.tolist(),'native':float(b[tuple(ix)]),'gpu':float(a[tuple(ix)])} for ix in indices[:8]]}
print(json.dumps({'shape':[nz,ny,nx],'turbines':nt,'model':model,'overlap':overlap,'dx_dt_factor':meta.tolist(),'status':cp.asnumpy(w.status).tolist(),'columns':((jy-1)*nx+ix-1).tolist(),'source_sha256':hashlib.sha256(raw).hexdigest(),'outputs':res,'wake_max_reduction':float(cp.max(w.ui-w.ul)), 'numerics':'strict' if ENABLED else 'fast','compiler':compile_receipt()},indent=2))
if args.debug:np.savez_compressed(args.debug,ui=cp.asnumpy(w.ui),ul=cp.asnumpy(w.ul),**actual,**{'native_'+k:v for k,v in expected.items()})

assert args.allow_differences or all(v['different']==0 for v in res.values()), 'native module replay differs'
