#!/usr/bin/env python3
"""Replay actual WRF gsfcgce call captures and compare every output word.

Use capture_real.py on an owned WRF source copy before a strict native
forecast. This tool never substitutes synthetic columns or repairs the
native output. CUDA events measure the two physics stages separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

IN3 = 'th qv qc qr qi qs qg rho pii p z dz8w'.split()
IN2 = 'ht rainnc snownc graupelnc'.split()
OUT3 = 'th qv qc qr qi qs qg refl'.split()
OUT2 = 'rainnc rainncv snownc snowncv graupelnc graupelncv sr'.split()

def decode(path, three, two, nx, nz):
    raw=np.fromfile(path,dtype='<f4')
    assert raw.size == len(three)*nx*nz+len(two)*nx,(path,raw.size)
    fields={}
    offset=0
    for name in three:
        fields[name]=raw[offset:offset+nx*nz].reshape(nz,1,nx).copy()
        offset+=nx*nz
    for name in two:
        fields[name]=raw[offset:offset+nx].reshape(1,nx).copy()
        offset+=nx
    return fields

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures',type=Path)
    parser.add_argument('receipt',type=Path)
    args=parser.parse_args()
    import cupy as cp
    from woof import wrf_exact
    from woof.core import mp_column_schemes as schemes
    row=schemes.MP_COLUMN_SCHEMES[97]
    result={'strict':wrf_exact.ENABLED,'environment':os.environ.get('GPUWM_WRF_EXACT'),
            'cases':[],'words':0,'differing_words':0,'nonfinite_outputs':0}
    meta_files=sorted(args.captures.glob('capture-*.meta'))
    if not meta_files:raise SystemExit('no native call captures found')
    for meta in meta_files:
        values=meta.read_text().split()
        nx,nz=map(int,values[:2])
        dt=float(values[2]); hail,ice=map(int,values[3:5])
        diag=values[5].upper().strip('.') in ('T','TRUE')
        radar=int(values[6]); grav,water,snow=map(float,values[7:10])
        assert np.float32(grav)==np.float32(9.81) and water==1000 and snow==100
        prefix=meta.with_suffix('')
        inputs=decode(prefix.with_suffix('.in'),IN3,IN2,nx,nz)
        before=decode(prefix.with_suffix('.pre'),OUT3,OUT2,nx,nz)
        expected=decode(prefix.with_suffix('.post'),OUT3,OUT2,nx,nz)
        device={name:cp.asarray(value) for name,value in inputs.items()}
        device.update({name:cp.asarray(value) for name,value in before.items()})
        # Compile and construct immutable tables before measuring physics.
        schemes.device_table(97,(hail,ice))
        from woof.core.kernels import get_kernel
        get_kernel(row.unit,row.entry)
        if diag and radar>0:
            # Initialize the shared radar tables and compile the specialization
            # before events measure the actual post-physics diagnostic.
            schemes.launch_refl(row,device['th'],device['pii'],device['p'],device['qv'],
                 device['qr'],device['qs'],device['qg'],cp.empty_like(device['refl']),
                 (hail,ice))
        begin,middle,end=[cp.cuda.Event() for _ in range(3)]
        begin.record()
        schemes.launch_column(row,device['th'],device,device['rho'],device['pii'],
             device['p'],device['z'],device['dz8w'],device['ht'],device['rainnc'],
             device['rainncv'],device['snownc'],device['snowncv'],device['graupelnc'],
             device['graupelncv'],device['sr'],np.float32(dt),(hail,ice))
        middle.record()
        if diag and radar>0:
            schemes.launch_refl(row,device['th'],device['pii'],device['p'],device['qv'],
                 device['qr'],device['qs'],device['qg'],device['refl'],(hail,ice))
        end.record(); end.synchronize()
        case={'capture':meta.name,'nx':nx,'nz':nz,'dt':dt,'hail':hail,'ice':ice,
              'diagflag':diag,'do_radar_ref':radar,'fields':{},
              'column_ms':cp.cuda.get_elapsed_time(begin,middle),
              'reflectivity_ms':cp.cuda.get_elapsed_time(middle,end),
              'sha256':{s:hashlib.sha256(prefix.with_suffix(s).read_bytes()).hexdigest()
                        for s in ('.in','.pre','.post','.meta')}}
        for name in OUT3+OUT2:
            actual=cp.asnumpy(device[name])
            ref=expected[name]
            mismatch=ref.view(np.uint32)!=actual.view(np.uint32)
            different=int(np.count_nonzero(mismatch))
            finite=np.isfinite(ref)&np.isfinite(actual)
            absolute=float(np.max(np.abs(ref[finite].astype(np.float64)-actual[finite]))) if finite.any() else None
            entry={'words':int(ref.size),'differing_words':different,
                   'max_absolute_finite':absolute,
                   'native_nonfinite':int(np.count_nonzero(~np.isfinite(ref))),
                   'gpu_nonfinite':int(np.count_nonzero(~np.isfinite(actual)))}
            if different:
                at=tuple(int(x) for x in np.argwhere(mismatch)[0])
                entry['first']={'index':at,'native_word':int(ref.view(np.uint32)[at]),
                                'gpu_word':int(actual.view(np.uint32)[at])}
            case['fields'][name]=entry
            result['words']+=int(ref.size)
            result['differing_words']+=different
            result['nonfinite_outputs']+=entry['native_nonfinite']+entry['gpu_nonfinite']
        result['cases'].append(case)
    result['compile']=wrf_exact.compile_receipt()
    result['pass']=result['strict'] and result['differing_words']==0 and result['nonfinite_outputs']==0
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('cases','compile')}))
    return 0 if result['pass'] else 1

if __name__=='__main__':sys.exit(main())
