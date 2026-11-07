"""Replay instrument_real.py's disposable native stream on CUDA.

Run under GPUWM_NUMERICS=strict and the shared GPU owner helper. Only the
small JSON receipt is retained. The native input and output words come
from the same, unmodified-arithmetic KF_cup_CPS call in wrf.exe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from woof.core.kfcup_contract import (
    KFCUP_LEVEL_INPUTS, KFCUP_SURFACE_INPUTS, KFCUP_HELD_SCALARS,
    KFCUP_HELD_INTS, KFCUP_LEVEL_OUTPUTS,
)


def load_stream(path):
    raw = Path(path).read_bytes()
    endian = next((e for e in ('<', '>') if struct.unpack(e+'i', raw[:4])[0] == 766721), None)
    if endian is None:
        raise ValueError('not an instrument_real.py capture')
    words = np.frombuffer(raw, dtype=endian+'u4')
    header = words[:8].view(endian+'i4').astype(np.int32)
    magic, version, nz, nc, ktau, stepcu, bins, flags = map(int, header)
    if version != 1 or ktau != 1:
        raise ValueError('replay requires the captured first actual WRF call')
    params = words[8:14].view(endian+'f4').astype(np.float32)
    offset = 14

    def read(n, integer=False):
        nonlocal offset
        a = words[offset:offset+n].view(endian+('i4' if integer else 'f4'))
        offset += n
        if a.size != n:
            raise ValueError('truncated capture')
        return a.astype(np.int32 if integer else np.float32)

    b = dict(lev_in=np.empty((12,nz,nc),np.float32), sca_in=np.empty((5,nc),np.float32),
             kpbl=np.empty(nc,np.int32), held=np.empty((18,nc),np.float32),
             iheld=np.empty((2,nc),np.int32), lev_out=np.empty((19,nz,nc),np.float32))
    w = np.empty((nz+1,nc),np.float32)
    for col in range(nc):
        b['lev_in'][:,:,col] = read(12*nz).reshape(12,nz)
        b['sca_in'][:,col] = read(5)
        b['kpbl'][col] = read(1,True)[0]
        w[:,col] = read(nz+1)
        b['held'][:,col] = read(18)
        b['iheld'][:,col] = read(2,True)
        b['lev_out'][:,:,col] = read(19*nz).reshape(19,nz)
    expected = dict(w0avg=np.empty((nz,nc),np.float32), held=np.empty_like(b['held']),
                    iheld=np.empty_like(b['iheld']), lev_out=np.empty_like(b['lev_out']))
    for col in range(nc):
        expected['w0avg'][:,col] = read(nz)
        expected['held'][:,col] = read(18)
        expected['iheld'][:,col] = read(2,True)
        expected['lev_out'][:,:,col] = read(19*nz).reshape(19,nz)
    if offset != words.size:
        raise ValueError(f'capture has {words.size-offset} trailing words')
    kwargs = dict(zip(('dt','dx','thbinsize','rbinsize','mindeepfreq','minshallowfreq'), params))
    kwargs.update(ntst=stepcu,numbins=bins,flags=flags)
    return b,w,expected,kwargs,dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
        levels=nz,columns=nc,ktau=ktau,byte_order=endian)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path)
    p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    from woof.numerics import mode
    if mode() != 'strict':
        raise RuntimeError('real native replay requires GPUWM_NUMERICS=strict')
    import cupy as cp
    from woof.core.kf import launch_kfcup_columns
    from woof.core.kernels import get_kernel, module_source
    b,w,expected,kwargs,source=load_stream(args.capture)
    gpu={k:cp.asarray(a) for k,a in b.items()}
    avg=gpu['lev_in'][KFCUP_LEVEL_INPUTS.index('w0avg')]
    nz,nc=avg.shape
    get_kernel('kfcup','kfcup_trigger_history')(((avg.size+255)//256,),(256,),
        (avg,cp.asarray(w),np.float32(2*kwargs['ntst']),np.int32(nc),np.int32(nz)))
    launch_kfcup_columns(gpu,**kwargs)
    got={k:gpu[k].get() for k in ('held','iheld','lev_out')}
    got['w0avg']=avg.get()
    rows=[]
    for key,names in (('w0avg',('w0avg',)),('held',KFCUP_HELD_SCALARS),
                       ('iheld',KFCUP_HELD_INTS),('lev_out',KFCUP_LEVEL_OUTPUTS)):
        actual=got[key][None] if key=='w0avg' else got[key]
        target=expected[key][None] if key=='w0avg' else expected[key]
        for name,a,e in zip(names,actual,target):
            bad=a.view(np.uint32)!=e.view(np.uint32)
            row=dict(field=name,words=a.size,differing_words=int(bad.sum()))
            if bad.any():
                idx=tuple(np.argwhere(bad)[0])
                row['first_difference']=dict(index=list(map(int,idx)),
                    actual=f'{int(a.view(np.uint32)[idx]):08x}',
                    native=f'{int(e.view(np.uint32)[idx]):08x}')
            rows.append(row)
    receipt=dict(source=source,configuration={k:float(v) for k,v in kwargs.items()},
        numerics='strict',gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cupy=cp.__version__,
        kernel_sha256=hashlib.sha256(module_source('kfcup').encode()).hexdigest(),
        compared_words=sum(r['words'] for r in rows),
        differing_words=sum(r['differing_words'] for r in rows),fields=rows)
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='fields'},indent=2))
    return 1 if receipt['differing_words'] else 0


if __name__=='__main__':
    raise SystemExit(main())
