"""Replay real-input captures of module_irrigation.F as REAL32 words."""
from pathlib import Path
import argparse,json
import numpy as np
import cupy as cp
from woof.core.kernels import get_kernel
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[]
def launch(h,sc,fields):
 nx,ny,x0,y0=fields.pop('_shape');args=tuple(fields[n] for n in ('irrigation','phase','rainbl','channel','qr','rho','dz'))
 option,sh,nh,sj,ej,freq,ph=map(int,h)
 jul,amount,gmt,xtime,dt=map(np.float32,sc)
 get_kernel('irrigation','irrigation_column')(((nx*ny+63)//64,),(64,),(*args,np.int32(option),amount,*map(np.int32,(sh,nh,sj,ej,freq,ph)),jul,xtime,gmt,dt,*map(np.int32,(nx,ny,x0,y0,0))))
def add(path,got,ref):
 words=diff=0
 for n in got:
  v=cp.asnumpy(got[n]).view(np.uint32);r=np.asarray(ref[n]).view(np.uint32);words+=v.size;diff+=int((v!=r).sum())
 row={'capture':path.name,'words':words,'differing':diff};rows.append(row);print(json.dumps(row),flush=True)
for path in sorted((a.case/'captures').glob('irrigation-*.bin')):
 # 10 integer metadata, 8 real inputs, 2 real and 1 integer outputs per call.
 data=path.read_bytes();assert len(data)%84==0
 for start in range(0,len(data),84):
  h=np.frombuffer(data,'<i4',10,start).copy();v=np.frombuffer(data,'<f4',10,start+40).copy();phase=np.frombuffer(data,'<i4',1,start+80).copy()
  f={n:cp.zeros(1,np.float32) for n in ('irrigation','rainbl','channel','qr','rho','dz')};f['irrigation'][...]=v[1];f['rainbl'][...]=v[6];f['channel'][...]=v[7];f['phase']=cp.asarray(h[9:10]);f['_shape']=(1,1,int(h[1]),int(h[2]))
  launch([h[0],*h[3:9]],v[[0,2,3,4,5]],f)
  add(path,{'rainbl':f['rainbl'],'channel':f['channel'],'phase':f['phase']},{'rainbl':v[8:9],'channel':v[9:10],'phase':phase})
for path in sorted((a.case/'captures').glob('sprinkler-*-in.bin')):
 data=path.read_bytes();h=np.frombuffer(data,'<i4',10).copy();v=np.frombuffer(data,'<f4',5,40).copy();nx,ny,x0,y0=map(int,h[:4]);n=nx*ny;off=60
 def take(dtype):
  global off
  x=np.frombuffer(data,dtype,n,off).copy();off+=4*n;return x
 host={'irrigation':take('<f4'),'phase':take('<i4'),'qr':take('<f4'),'rho':take('<f4'),'dz':take('<f4')};assert off==len(data)
 f={n:cp.asarray(x) for n,x in host.items()};f.update(rainbl=cp.zeros(n,np.float32),channel=cp.zeros(n,np.float32),_shape=(nx,ny,x0,y0));launch([3,*h[4:]],v,f)
 out=path.with_name(path.name.replace('-in.bin','-out.bin')).read_bytes();ref={'qr':np.frombuffer(out,'<f4',n).copy(),'phase':np.frombuffer(out,'<i4',n,4*n).copy()};add(path,{n:f[n] for n in ref},ref)
a.out.write_text(json.dumps({'words':sum(r['words'] for r in rows),'differing':sum(r['differing'] for r in rows),'captures':rows},indent=2)+'\n')
