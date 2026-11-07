"""Replay only real native forecast stage captures; no manufactured states."""
from pathlib import Path
import argparse
import ctypes as C
import json,time,hashlib
import numpy as np
import cupy as cp
from woof.core.kernels import get_kernel
from woof.core.tmn_update import launch
FLOATS='xland glw gsw hfx qfx tsk ust emiss sst dtw sstsk tmn xice xicem xice_save vegfra albedo albbck embck albsi tsk_save snow snowc snowh lakemask tyr tyra tdly'.split()
def read(path):
 b=path.read_bytes();off=0
 def take(n,dtype):
  nonlocal off
  a=np.frombuffer(b,dtype=dtype,count=n,offset=off).copy();off+=a.nbytes;return a
 h=take(13,'<i4');p=take(4,'<f4');n,ns,nl=h[:3]
 f={k:take(n,'<f4') for k in FLOATS}
 for k in ('ivgtyp','isltyp'):f[k]=take(n,'<i4')
 for k in ('tslb','smois','sh2o'):f[k]=take(n*ns,'<f4').reshape(ns,n)
 f['tlag']=take(n*nl,'<f4').reshape(nl,n)
 f['nday']=take(1,'<f4');f['nyear']=take(1,'<i4')
 assert off==len(b),(off,len(b));return h,p,f
def gpu_stage(stage,h,p,f):
 n,ns,nl,yr,step,ss,skin,tmn,frac,alb,water,ice,lake=map(int,h);dt,julian,stbolt,icealb=p
 if stage==1 and ss:
  names='sst lakemask xice xicem xice_save xland ivgtyp isltyp vegfra tmn albedo albbck emiss embck albsi tsk tsk_save snow snowc snowh tslb smois sh2o'.split()
  get_kernel('surface_analysis','surface_update_column')(((n+255)//256,),(256,),
   (*(f[k] for k in names),*map(np.int32,(lake,frac,water,ice,alb)),np.float32(icealb),np.int32(ns),np.int32(n)))
 if stage==2 and skin:
  names='xland glw gsw hfx qfx tsk ust emiss sst dtw sstsk tslb'.split()
  get_kernel('sstskin','sstskin_column')(((n+127)//128,),(128,),
   (*(f[k] for k in names),np.float32(dt),np.float32(stbolt),np.int32(ss),np.int32(n)))
 if stage==3 and tmn:
  g={k:v.reshape((nl,1,n) if k=='tlag' else (1,) if k in ('nday','nyear') else (1,n)) for k,v in f.items() if k in ('tmn','tlag','tyr','tyra','tdly','nday','nyear')}
  launch(g,f['tsk'].reshape(1,n),julian=julian,dt=dt,year=yr)


def compare(gpu,want):
 diff={};words=0
 for k,v in gpu.items():
  got=cp.asnumpy(v);a=got.view(np.uint32).ravel();b=want[k].view(np.uint32).ravel();words+=len(a)
  ix=np.flatnonzero(a!=b)
  if len(ix):diff[k]={'words':len(ix),'first':int(ix[0]),'got_word':int(a[ix[0]]),'want_word':int(b[ix[0]])}
 return dict(words=words,differing_words=sum(v['words'] for v in diff.values()),differences=diff)


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--captures',type=Path,required=True)
 ap.add_argument('--oracle',type=Path,required=True)
 ap.add_argument('--receipt',type=Path,required=True)
 args=ap.parse_args()
 from woof.wrf_exact import ENABLED
 if not ENABLED:raise SystemExit('GPUWM_WRF_EXACT=1 is required for this proof')
 lib=C.CDLL(str(args.oracle.resolve()))
 oracle=lib.surface_stage
 oracle.argtypes=[C.c_int]*3+[C.c_void_p]*9+[C.c_int]
 results=[]
 for src in sorted(args.captures.glob('*-pre.bin')):
  h,p,host=read(src);_,_,want=read(src.with_name(src.name.replace('-pre','-post')))
  n,ns,nl=map(int,h[:3]);f={k:cp.asarray(v) for k,v in host.items()}
  packed=np.stack([host[k] for k in FLOATS]);soil=np.stack([host[k] for k in ('tslb','smois','sh2o')])
  native=dict(host)
  for i,k in enumerate(FLOATS):native[k]=packed[i]
  for i,k in enumerate(('tslb','smois','sh2o')):native[k]=soil[i]
  buffers=[h,p,packed,native['ivgtyp'],native['isltyp'],soil,native['tlag'],native['nday'],native['nyear']]
  stages=[]
  for stage,name in enumerate(('sst_update','sst_skin','tmn_update'),1):
   oracle(n,ns,nl,*(v.ctypes.data for v in buffers),stage)
   gpu_stage(stage,h,p,f)
   stages.append(dict(stage=name,**compare(f,native)))
  results.append(dict(capture=src.name,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),step=int(h[4]),stages=stages,native_forecast_witness=compare(f,want)))
 total=sum(s['words'] for r in results for s in r['stages'])
 difference=sum(s['differing_words'] for r in results for s in r['stages'])
 witness=sum(r['native_forecast_witness']['differing_words'] for r in results)
 result=dict(schema='sst-surface-native-module-replay-v2',strict=True,oracle_sha256=hashlib.sha256(args.oracle.read_bytes()).hexdigest(),captures=results,words=total,differing_words=difference,native_forecast_witness_differing_words=witness)
 args.receipt.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(words=total,differing_words=difference,native_forecast_witness_differing_words=witness)))
 return bool(difference or witness or not results)

if __name__=='__main__':raise SystemExit(main())
