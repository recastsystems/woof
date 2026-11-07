#!/usr/bin/env python3
"""Locate the first changed routine snapshot in the lowering debug trace."""
import argparse,json,re
from pathlib import Path
import numpy as np
from lower import split
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('native',type=Path);p.add_argument('host',type=Path)
a=p.parse_args();meta=json.loads((a.work/'trace-fields.json').read_text())
def read(path,ff,nchem):
 raw=path.read_bytes();off=0;result={};sizes=dict(kts=1,kte=49,nchem=nchem,ndvel=nchem)
 for n,f in ff.items():
  if f['type']=='character':continue
  count=1
  if f['dims']:
   for d in split(f['dims']):
    lo,hi=d.split(':') if ':' in d else ('1',d)
    def ev(expr):return eval(expr,{'__builtins__':{}},sizes)
    count*=ev(hi)-ev(lo)+1
  v=np.frombuffer(raw,dtype='<f4' if f['type']=='real' else '<i4',count=count,offset=off)
  off+=4*count;result[n]=v
  if not f['dims'] and f['type']=='integer':sizes[n]=int(v[0])
 assert off==len(raw),(path,off,len(raw))
 return result
report={}
for path in sorted(a.native.glob('*')):
 other=a.host/path.name
 if not other.exists():continue
 name=path.name.split('-')[0]
 x,y=read(path,meta[name],2),read(other,meta[name],1)
 diff={}
 for n in x:
  if x[n].shape!=y[n].shape or n in ('nchem','ndvel','chem','chem1','s_awchem1'):continue
  mask=x[n].view('u4')!=y[n].view('u4')
  if mask.any():
   k=int(np.flatnonzero(mask)[0]);diff[n]=dict(count=int(mask.sum()),first=k,native=float(x[n][k]),host=float(y[n][k]))
 if diff:report[path.name]=diff
print(json.dumps(report,indent=2))
