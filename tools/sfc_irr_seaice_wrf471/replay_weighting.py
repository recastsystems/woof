"""Replay the exact native pre/post-LSM fractional transformations."""
from pathlib import Path
import argparse
import json
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('case',type=Path)
p.add_argument('--layout',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
import cupy as cp
from woof.core.noah_fractional_seaice import before_lsm,after_lsm
names=json.loads((a.layout/'weighting-layout.json').read_text())['f2']
def read(path):
    raw=path.read_bytes()
    h=np.frombuffer(raw,'<i4',6).copy()
    assert np.frombuffer(raw,'<f4',1,24)[0]==np.float32(.02)
    nx,ny=map(int,h[:2])
    v=np.frombuffer(raw,'<f4',offset=28).reshape(len(names),ny,nx)
    return h,dict(zip(names,v))
rows=[]
for path in sorted((a.case/'captures').glob('weighting-*-in.bin')):
    h,host=read(path)
    _,ref=read(path.with_name(path.name.replace('-in.bin','-out.bin')))
    dev={n:cp.asarray(v) for n,v in host.items()}
    if h[4]==0:before_lsm(dev,int(h[2]),surface_due=bool(h[5]))
    else:after_lsm(dev,surface_due=bool(h[5]))
    fields={};words=diff=0
    for n,r in ref.items():
        value=cp.asnumpy(dev[n]);left,right=value.view(np.uint32),r.view(np.uint32)
        count=int((left!=right).sum());words+=left.size;diff+=count
        if count:
            pos=tuple(np.argwhere(left!=right)[0])
            fields[n]=dict(differing=count,first_index=list(map(int,pos)),
                gpu=float(value[pos]),native=float(r[pos]),
                gpu_word=hex(int(left[pos])),native_word=hex(int(right[pos])))
    row=dict(capture=path.name,words=words,differing=diff,fields=fields)
    rows.append(row);print(json.dumps(row),flush=True)
a.out.write_text(json.dumps(dict(words=sum(r['words'] for r in rows),
    differing=sum(r['differing'] for r in rows),captures=rows),indent=2)+'\n')
