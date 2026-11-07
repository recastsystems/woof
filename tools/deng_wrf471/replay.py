"""Replay full native Deng driver state, comparing every returned word."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read_capture(path,layout):
    payload=path.read_bytes();pos=0
    def take(n,typ='<f4'):
        nonlocal pos
        value=np.frombuffer(payload,typ,n,pos).copy();pos+=value.nbytes;return value
    step,its,ite,jts,jte,kts,kte=map(int,take(7,'<i4'))
    nx,ny,nz=ite-its+1,jte-jts+1,kte-kts+1
    tables=take(110423)
    def field(m):
        dtype='<f4' if m['type']=='float' else '<i4'
        name=m['name'];dims=m['dims']
        if not dims:
            return take(1,dtype)[0]
        if name=='dsigma':return take(nz+1,dtype)
        if len(dims)==2:return take(nx*ny,dtype).reshape(ny,nx)
        nk=100 if name=='ainckfsa' else nz+1 if name in ('w','z_at_w') else nz
        return np.ascontiguousarray(take(nx*nk*ny,dtype).reshape(ny,nk,nx).transpose(1,0,2))
    before={m['name']:field(m) for m in layout}
    after={m['name']:field(m) for m in layout if m['intent'] in ('out','inout')}
    assert pos==len(payload),(pos,len(payload))
    return (nz,ny,nx),step,tables,before,after


def replay(path,layout):
    import cupy as cp
    from woof.core.kernels import load_module,kernel_local_size_bytes
    shape,step,tables,before,after=read_capture(path,layout)
    nz,ny,nx=shape;ncol=ny*nx
    args=[];arrays={}
    for m in layout:
        name=m['name'];a=before[name]
        if isinstance(a,np.ndarray):a=arrays[name]=cp.asarray(a)
        elif m['type']=='bool':a=np.bool_(a)
        args.append(a)
    slots=512*nz+8192
    workspace=cp.empty(slots*ncol,dtype=cp.float32);status=cp.zeros(1,dtype=cp.int32)
    fn=load_module('deng').get_function('deng_column')
    fn(((ncol+63)//64,),(64,), (np.int32(nz),np.int32(ncol),workspace,np.int32(slots),cp.asarray(tables),*args,status))
    flag=int(status.get()[0]);assert flag==0,flag
    result={}
    for name,want in after.items():
        got=cp.asnumpy(arrays[name]);mask=got.view('u4')!=want.view('u4')
        result[name]=dict(words=int(want.size),differing=int(mask.sum()),active_native=int(np.count_nonzero(want)),
            examples=[dict(index=idx.tolist(),native=float(want[tuple(idx)]),gpu=float(got[tuple(idx)]),
                           native_word=int(want.view('u4')[tuple(idx)]),gpu_word=int(got.view('u4')[tuple(idx)])) for idx in np.argwhere(mask)[:5]])
    row=dict(capture=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=step,shape=shape,
             frame_bytes=kernel_local_size_bytes(fn),outputs=result,
             words=sum(v['words'] for v in result.values()),differing=sum(v['differing'] for v in result.values()))
    print(json.dumps({k:v for k,v in row.items() if k!='outputs'}),flush=True)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    layout=json.loads((a.directory.parent/'capture-layout.json').read_text())
    rows=[replay(f,layout) for f in sorted(a.directory.glob('*.bin'))]
    a.output.write_text(json.dumps(rows,indent=2)+'\n')
    raise SystemExit(int(not rows or any(r['differing'] for r in rows)))
