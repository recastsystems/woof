"""Replay instrumented, real native WRF arrays. Offline proof only."""
from pathlib import Path
import sys,json,hashlib,os,argparse
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('capture',type=Path)
parser.add_argument('receipt',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
os.environ['GPUWM_WRF_EXACT']='1'
import cupy as cp
from woof import wrf_exact
wrf_exact.install()
source=(root/'woof/fddaobs/kernels.cu').read_text()
module=cp.RawModule(code=source,options=('--std=c++17',))

def record(values,types):
 return np.array(tuple(values),dtype=[('v'+str(i),t) for i,t in enumerate(types)])[()]
class Reader:
 def __init__(self,p): self.p=p; self.f=p.open('rb')
 def a(self,n,t='<f4'):
  a=np.fromfile(self.f,dtype=t,count=n)
  if len(a)!=n:raise ValueError((self.p,self.f.tell(),n,len(a)))
  return a
 def end(self): assert not self.f.read(1),self.p

def diff(name,a,b):
 aw=np.asarray(a).view('u4');bw=np.asarray(b).view('u4');ix=np.flatnonzero(aw!=bw)
 return {'name':name,'words':int(a.size),'differing_words':int(ix.size),'first':[{'offset':int(i),'native':float(a[i]),'cuda':float(b[i]),'native_word':int(aw[i]),'cuda_word':int(bw[i])} for i in ix[:12]]}

def errob(p):
 r=Reader(p);g=r.a(18,'<i4');nc,nv,ne,wind,temp,mois,nest=r.a(7,'<i4');t0,rcp=r.a(2)
 nx,nk,ny=[int(g[b]-g[a]+1) for a,b in [(6,7),(10,11),(8,9)]];size=nx*nk*ny;xy=nx*ny
 arrays=[cp.asarray(r.a(size)) for _ in range(7)]
 ratios=[cp.asarray(r.a(xy)) for _ in range(3)];kp=cp.asarray(r.a(xy,'<i4'))
 rio,rjo,rko=[cp.asarray(r.a(nc)) for _ in range(3)];obs=cp.asarray(r.a(nv*nc));r.end()
 err=cp.zeros(nc*ne,dtype='f4');up=cp.zeros(xy,dtype='f4');vp=cp.zeros_like(up);gg=record(g,['<i4']*18)
 module.get_function('fddaobs_ratios')(((nx+15)//16,(ny+15)//16),(16,16),(ratios[0],ratios[1],up,vp,gg))
 module.get_function('fddaobs_errob')(((int(nc)+127)//128,),(128,),(*arrays,up,vp,ratios[2],kp,rio,rjo,rko,obs,err,np.int32(nc),np.int32(nv),np.int32(ne),wind,temp,mois,t0,rcp,gg,np.uint64(0)))
 o=Reader(p.with_name(p.name.replace('-in.bin','-out.bin')))
 res=[diff(p.stem+':'+name,o.a(n),a.get()) for name,n,a in [('rko',nc,rko),('varobs',nv*nc,obs),('errf',ne*nc,err)]];o.end()
 return res

def nudob(p):
 r=Reader(p);g=r.a(18,'<i4');ints=r.a(14,'<i4');nc,nv,ne,ivar,nest,ktau,ktaur,restart,interval,pbl,horiz,vert,neg,j=ints
 fs=r.a(16);xt,dt,rin,dx,win,sff,sfr,rmin,rmax,pfree,dcon,tfaci,giv,git,giq,gap=fs
 nx=int(g[7]-g[6]+1);nk=int(g[11]-g[10]+1);size=nx*nk
 aten0=r.a(size);aten=cp.asarray(aten0);sav=cp.asarray(r.a(nv*size))
 mu,c1,c2,mx,my=[cp.asarray(r.a(n)) for n in [nx,nk,nk,nx,nx]]
 rio,rjo,rko,time=[cp.asarray(r.a(nc)) for _ in range(4)]
 obs=cp.asarray(r.a(nc*nv));err=cp.asarray(r.a(nc*ne));nlev,lev=[cp.asarray(r.a(nc)) for _ in range(2)]
 pb,pp=[cp.asarray(r.a(size)) for _ in range(2)];kp=cp.asarray(r.a(nx,'<i4'))
 def vif():
  v=np.zeros(nx,dtype='f4');v[int(g[12]-g[6]):int(g[13]-g[6]+1)]=r.a(int(g[13]-g[12]+1));return cp.asarray(v)
 v1,v2=vif(),vif();terr=cp.asarray(r.a(nx));z,qv=[cp.asarray(r.a(size)) for _ in range(2)];r.end()
 g[8]=g[9]=g[14]=g[15]=j;gg=record(g,['<i4']*18)
 coefficient=giv if ivar<=2 else git if ivar==3 else giq
 c=record([nest,ktau,ktaur,restart,interval,pbl,horiz,vert,neg,xt,dt,rin,dx,win,sff,sfr,rmin,rmax,pfree,dcon,tfaci,coefficient,gap],['<i4']*9+['<f4']*14)
 wt=cp.zeros_like(aten);w2=cp.zeros_like(aten)
 top=np.full(nx,int(g[17]),dtype='i4')
 zh=z.get().reshape((nk,nx));terrain=terr.get();lo=v1.get();hi=v2.get()
 for i in range(nx):
  for k in range(nk):
   wv=max(np.float32(0),min(np.float32(1),np.float32(1)+(lo[i]-zh[k,i]+terrain[i])*(np.float32(1)/max(np.float32(1),hi[i]-lo[i]))))
   if wv<=0:top[i]=k;break
 top=cp.asarray(top)
 module.get_function('fddaobs_nudob_rows')(*(((1,), (32,)) if neg else (((nx+31)//32,1,(nk+3)//4),(32,1,4))),(aten,sav,wt,w2,mu,c1,c2,mx,my,rio,rjo,rko,time,obs,err,nlev,lev,pb,pp,kp,v1,v2,terr,z,qv,nc,nv,ne,ivar,c,gg,np.uint64(0),top))
 o=Reader(p.with_name(p.name.replace('-in.bin','-out.bin')))
 res=[diff(p.stem+':'+name,o.a(n),a.get()) for name,n,a in [('aten',size,aten),('savwt',nv*size,sav)]];res[0]['changed_tendency_words']=int(np.count_nonzero(aten0.view('u4')!=aten.get().view('u4')));o.end();return res

results=[];paths=sorted(args.capture.glob('*-in.bin'))
if not paths:raise RuntimeError('no real native captures')
for p in paths:
 res=errob(p) if 'errob' in p.name else nudob(p);results.extend(res)
 for x in res: print(json.dumps(x),flush=True)
out={'native_inputs':[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths], 'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'results':results,'compared_words':sum(x['words'] for x in results),'differing_words':sum(x['differing_words'] for x in results),'strict_compile':wrf_exact.compile_receipt()}
args.receipt.write_text(json.dumps(out,indent=2)+'\n')
print('TOTAL',out['compared_words'],out['differing_words'])

if out['differing_words']:raise SystemExit(1)
