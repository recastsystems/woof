"""Replay retained IDEALSCMSFCLAY real native entry and exit streams."""
from pathlib import Path
import json,sys
import numpy as np
import cupy as cp
from woof.core.idealscmsfclay import launch, OUTPUTS
from woof.core.physics_inventory import SCM_SURFACE_FORCING_FIELDS
surfaces=('hfx','lh','qfx','tsk','flhc','exch_temf','flqc','ust','chs2','cqs2','chs','cpm','qgh','qsfc','u10','v10','th2','t2','q2')
report=[]
for p in sorted(Path(sys.argv[1]).glob('*.input')):
 raw=p.read_bytes(); its,ite,jts,jte,step=map(int,np.frombuffer(raw,'<i4',count=5)); ns=(ite-its+1)*(jte-jts+1)
 dt,cp_air,xlv=np.frombuffer(raw,'<f4',count=3,offset=20);offset=32
 def take(n):
  global offset
  a=cp.asarray(np.frombuffer(raw,'<f4',count=n,offset=offset).copy());offset+=n*4;return a
 forcing={n:take(1) for n in SCM_SURFACE_FORCING_FIELDS}
 th,pi,rho,mavail=(take(ns) for _ in range(4))
 a={n:take(ns) for n in surfaces};assert offset==len(raw)
 launch(th,pi,rho,mavail,forcing,{n:a[n] for n in OUTPUTS},dt=float(dt),cp_air=float(cp_air),xlv=float(xlv))
 expected=p.with_suffix('.output').read_bytes();offset=0;row={'capture':p.name,'step':step,'words':0,'different':0,'fields':{}}
 for name,arr in list(forcing.items())+list(a.items()):
  actual=cp.asnumpy(arr).ravel().view('u4');ref=np.frombuffer(expected,'<u4',count=actual.size,offset=offset);offset+=actual.nbytes
  bad=actual!=ref;info={'words':int(actual.size),'different':int(bad.sum())}
  if bad.any():
   info['first']=[{'index':int(i),'native':f'{ref[i]:08x}','gpu':f'{actual[i]:08x}'} for i in np.flatnonzero(bad)[:8]]
  row['fields'][name]=info;row['words']+=info['words'];row['different']+=info['different']
 assert offset==len(expected);report.append(row);print(json.dumps(row),flush=True)
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(int(any(r['different'] for r in report)))
