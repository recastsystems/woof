"""Compare a production Rust pack with captured native reader inputs.

The selected native window must be a prefix of the packed admitted rows;
retain the native run's original clock and exact projection in its request.
"""
from pathlib import Path
import argparse,json
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('pack',type=Path)
parser.add_argument('native',type=Path)
parser.add_argument('receipt',type=Path)
args=parser.parse_args()
p=args.pack.read_bytes();n=int.from_bytes(p[8:12],'little');meta=json.loads(p[12:12+n]);a=np.frombuffer(p,dtype='<f4',offset=12+n);count=meta['rows']
f=args.native
with f.open('rb') as r:
 g=np.fromfile(r,'<i4',18);nc,nv,ne,*_=np.fromfile(r,'<i4',7);r.read(8)
 nx=g[7]-g[6]+1;ny=g[9]-g[8]+1;nz=g[11]-g[10]+1
 r.seek((7*nx*ny*nz+4*nx*ny)*4,1)
 native=[np.fromfile(r,'<f4',nc) for _ in range(3)]+[np.fromfile(r,'<f4',nc*nv)]
 assert not r.read(1)
rust=[a[v*count:v*count+nc] for v in range(3)]+[a[8*count:8*count+nc*nv]]
results=[]
for name,x,y in zip(('rio','rjo','rko','varobs'),native,rust):
 ix=np.flatnonzero(x.view('u4')!=y.view('u4'))
 results.append(dict(name=name,words=x.size,differing_words=ix.size,first=[dict(index=int(i),native=float(x[i]),rust=float(y[i])) for i in ix[:8]]))
results=json.loads(json.dumps(results,default=int));out=dict(metadata=meta,results=results,compared_words=sum(r['words'] for r in results),differing_words=sum(r['differing_words'] for r in results))
args.receipt.write_text(json.dumps(out,indent=2)+'\n');print(out['compared_words'],out['differing_words']);assert not out['differing_words']
