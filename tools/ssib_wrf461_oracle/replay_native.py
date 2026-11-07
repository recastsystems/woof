"""Replay CASE-arm records captured inside a running native WRF executable."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from woof.core import ssib

def read_capture(path):
    raw=Path(path).read_bytes()
    h=np.frombuffer(raw,'<i4',10)
    magic,ncol,nz,step=map(int,h[:4])
    if magic!=1397967170 or ncol<1 or nz<1:
        raise ValueError(f'{path}: invalid native SSiB capture header')
    dtype=np.dtype([('ij','<i4',(2,)),('static','<f4',(6,)),
                    ('state','<f4',(111,)),('forcing','<f4',(24+nz,))])
    if len(raw)!=40+ncol*dtype.itemsize:
        raise ValueError(f'{path}: truncated capture')
    return h,np.frombuffer(raw,dtype,ncol,40)

def main():
    import cupy as cp
    from woof import wrf_exact
    ap=argparse.ArgumentParser()
    ap.add_argument('directory',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    rows=[]
    for path in sorted(args.directory.glob('ssib-*-in.bin')):
        output=path.with_name(path.name.replace('-in.bin','-out.bin'))
        if not output.is_file():continue
        h,a=read_capture(path);ho,b=read_capture(output)
        np.testing.assert_array_equal(h,ho)
        np.testing.assert_array_equal(a['ij'],b['ij'])
        static,state,forcing=(cp.asarray(np.ascontiguousarray(a[n].T))
                              for n in ('static','state','forcing'))
        errors=ssib.step(static,state,forcing,nz=int(h[2]),itimestep=int(h[3]),fixes=0)
        actual=state.get().T.copy();expected=b['state'].copy()
        bad=actual.view(np.uint32)!=expected.view(np.uint32)
        fields={}
        for k,name in enumerate(ssib.STATE):
            if not bad[:,k].any():continue
            idx=np.flatnonzero(bad[:,k])
            fields[name]={'different_words':int(idx.size),'first':[
                {'ij':a['ij'][i].tolist(),
                 'actual_hex':f'{actual.view(np.uint32)[i,k]:08x}',
                 'native_hex':f'{expected.view(np.uint32)[i,k]:08x}',
                 'actual':str(actual[i,k]),'native':str(expected[i,k])}
                 for i in idx[:4]]}
        rows.append(dict(input=path.name,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            native_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            step=int(h[3]),columns=int(h[1]),nz=int(h[2]),words=int(actual.size),
            different_words=int(bad.sum()),errors=errors,fields=fields))
    receipt=dict(strict=wrf_exact.ENABLED,calls=len(rows),
        words=sum(r['words'] for r in rows),different_words=sum(r['different_words'] for r in rows),
        native='WRF v4.6.1 actual module_surface_driver CASE (SSIBSCHEME)',rows=rows)
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}))
    if not rows:raise SystemExit('No completed native calls found')
    if receipt['different_words'] or any(r['errors'] & ~ssib.ERR_ICASE_UNSET for r in rows):
        raise SystemExit('Native SSiB replay differs or reports a fatal status')
if __name__=='__main__':main()
