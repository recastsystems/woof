"""Replay only native-captured Fitch inputs. No generated weather columns."""
from pathlib import Path
import argparse,json,hashlib,os
import numpy as np
import cupy as cp
from woof.wrf_exact import ENABLED, compile_receipt
from woof.core.kernels import get_kernel
ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path);ap.add_argument('--debug',type=Path);ap.add_argument('--allow-differences',action='store_true',help='diagnostic fast-mode profile only');args=ap.parse_args()
p=args.capture;raw=(p/'fitch-native.in').read_bytes();at=0
def read(n,kind='f4'):
 global at
 a=np.frombuffer(raw,dtype='>'+kind,count=n,offset=at).astype(kind);at+=n*4;return a
nx,ny,nz,nt,induction=map(int,read(5,'i4'));nc=nx*ny;meta=read(3)
diam,hub=[read(nt) for _ in range(2)];ix,jy,nkind=[read(nt,'i4') for _ in range(3)]
stc,stc2,cutin,cutout,npower=[read(nt) for _ in range(5)]
curves=[read(10000).reshape(100,100) for _ in range(3)]
def vol():return read(nx*nz*ny).reshape(ny,nz,nx).transpose(1,0,2).copy()
u,v,dz=[vol() for _ in range(3)];zw=read(nx*(nz+1)*ny).reshape(ny,nz+1,nx).transpose(1,0,2).copy();du,dv,qke=[vol() for _ in range(3)]
assert at==len(raw),(at,len(raw))
out=np.frombuffer((p/'fitch-native.out').read_bytes(),dtype='>f4').astype('f4')
expected={}
for i,name in enumerate(['du','dv','qke']):expected[name]=out[i*nz*nc:(i+1)*nz*nc].reshape(ny,nz,nx).transpose(1,0,2).copy()
for i,name in enumerate(['POWER','THRUST']):expected[name]=out[3*nz*nc+i*nc:3*nz*nc+(i+1)*nc].reshape(ny,nx)
gpu=lambda a:cp.asarray(np.ascontiguousarray(a))
col=(jy-1)*nx+ix-1;ids=np.argsort(col,kind='stable').astype('i4');counts=np.bincount(col,minlength=nc).astype('i4');ptr=np.r_[np.int32(0),np.cumsum(counts,dtype='i4')];cols=np.flatnonzero(counts).astype('i4')
# Native tables are type-major; the CUDA resident decoder expands type refs once.
ws,tc,pw=[np.stack([c[:,k-1] for k in nkind],axis=1).copy() for c in curves]
# npower is nominal MW, not the number of samples. Native stores the
# latter only locally during init and retains cutout as the final speed.
nv=np.asarray([np.flatnonzero(ws[:,t]==cutout[t])[0]+1 for t in range(nt)],'i4')
gdu,gdv,gq=[gpu(a) for a in (du,dv,qke)];adv=gq.copy();power=cp.zeros((ny,nx),'f4');thrust=cp.zeros_like(power);rate=cp.zeros_like(gq);inc=cp.zeros_like(gq);hout=cp.zeros((nt,10),'f4');tp=cp.zeros(nt,'f4')
block=max(32,1<<(nz-1).bit_length());get_kernel('wind_mav','fitch_induction')((len(cols),),(block,),
(gpu(cols),gpu(ptr),gpu(ids),gpu(diam),gpu(hub),gpu(nv),gpu(ws),gpu(tc),gpu(pw),gpu(stc),gpu(stc2),gpu(u),gpu(v),gpu(dz),gpu(zw),gdu,gdv,gq,adv,power,thrust,rate,inc,hout,tp,np.int32(nc),np.int32(nz),np.int32(nt),*meta),shared_mem=nz*4)
cp.cuda.get_current_stream().synchronize();actual=dict(du=cp.asnumpy(gdu),dv=cp.asnumpy(gdv),qke=cp.asnumpy(gq),POWER=cp.asnumpy(power),THRUST=cp.asnumpy(thrust));res={}
for name,b in expected.items():
 a=actual[name];neq=a.view('u4')!=b.view('u4');indices=np.argwhere(neq);res[name]=dict(words=a.size,different=int(neq.sum()),max_absolute=float(np.nanmax(abs(a-b))),first=[dict(index=i.tolist(),native=float(b[tuple(i)]),gpu=float(a[tuple(i)])) for i in indices[:8]])
print(json.dumps(dict(shape=[nz,ny,nx],turbines=nt,induction=induction,source_sha256=hashlib.sha256(raw).hexdigest(),numerics='strict' if ENABLED else 'fast',compiler=compile_receipt(),outputs=res),indent=2))
if args.debug:np.savez_compressed(args.debug,**actual,**{'native_'+k:v for k,v in expected.items()})

assert args.allow_differences or all(v['different']==0 for v in res.values()), 'native module replay differs'
