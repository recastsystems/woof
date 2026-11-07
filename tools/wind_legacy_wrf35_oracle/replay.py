#!/usr/bin/env python3
"""Replay real captured WRF turbine_drag inputs through the production reader."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import numpy as np


def load_capture(path):
    raw=(path/'legacy-native.in').read_bytes(); at=0
    def read(n,kind='f4'):
        nonlocal at
        a=np.frombuffer(raw,dtype='>'+kind,count=n,offset=at).astype(kind)
        at+=4*n
        return a
    nx,ny,nz,nt=map(int,read(4,'i4')); nc=nx*ny
    meta=read(3); ids=[]; turbines=[]
    for _ in range(nt):
        ids.append(int(read(1,'i4')[0])); turbines.append(read(10))
    def volume(levels=nz):
        return read(nx*ny*levels).reshape(ny,levels,nx).transpose(1,0,2).copy()
    data=dict(shape=(nz,ny,nx),meta=meta,grid_ids=np.asarray(ids),
              turbines=np.asarray(turbines),input_sha256=hashlib.sha256(raw).hexdigest())
    for name in ('u','v','dz'):data[name]=volume()
    data['phb']=volume(nz+1)
    for name in ('du','dv','qke','qke_adv'):data[name]=volume()
    for name in ('XLAT_U','XLONG_U'):data[name]=read((nx+1)*ny).reshape(ny,nx+1)
    for name in ('XLAT_V','XLONG_V'):data[name]=read(nx*(ny+1)).reshape(ny+1,nx)
    assert at==len(raw),(at,len(raw))
    out=(path/'legacy-native.out').read_bytes()
    words=np.frombuffer(out,dtype='>f4').astype('f4')
    assert len(words)==4*nz*nc
    data['output_sha256']=hashlib.sha256(out).hexdigest()
    data['expected']={name:words[i*nz*nc:(i+1)*nz*nc].reshape(ny,nz,nx).transpose(1,0,2).copy()
                      for i,name in enumerate(('du','dv','qke','qke_adv'))}
    return data


def replay(path):
    import cupy as cp
    from woof.core.wind_legacy import LegacyFarm
    from woof.wrf_exact import ENABLED,compile_receipt
    x=load_capture(path);nz,ny,nx=x['shape']
    cfg=SimpleNamespace(windturbines_spec=str(path/'windspec.in'),grid_id=1,
                        dx=float(x['meta'][0]),dt=float(x['meta'][1]),bl_mynn_tkeadvect=True)
    farm=LegacyFarm(cfg,x['shape'],static={k:x[k] for k in ('XLAT_U','XLONG_U','XLAT_V')},phb=x['phb'])
    gpu=lambda a:cp.asarray(np.ascontiguousarray(a))
    atmosphere={k:gpu(x[k]) for k in ('u','v','dz')}
    fields={k:gpu(x[k]) for k in ('qke','qke_adv')}
    rates={k:gpu(x[k]) for k in ('du','dv')}
    farm.apply_mynn(atmosphere,fields,rates)
    actual={k:cp.asnumpy(v) for k,v in (rates|fields).items()}
    results={}
    for name,b in x['expected'].items():
        a=actual[name]; neq=a.view('u4')!=b.view('u4')
        indices=np.argwhere(neq)
        results[name]=dict(words=a.size,different=int(neq.sum()),
            max_absolute=float(np.nanmax(abs(a-b))),
            first=[dict(index=i.tolist(),native=float(b[tuple(i)]),gpu=float(a[tuple(i)])) for i in indices[:8]])
    rows=x['turbines']; active=(x['grid_ids']==1)&(rows[:,2]>=1)&(rows[:,2]<=nx)&(rows[:,3]>=1)&(rows[:,3]<=ny)
    native_columns=((rows[active,3]-1)*nx+rows[active,2]-1).astype('i4')
    coordinate_differences=int(np.count_nonzero(cp.asnumpy(farm.col)!=native_columns))
    coefficient_differences=int(np.count_nonzero(cp.asnumpy(farm.params).view('u4')!=rows[active,4:].copy().view('u4')))
    result=dict(schema='wind-legacy-native-replay.v1',shape=x['shape'],
        input_sha256=x['input_sha256'],output_sha256=x['output_sha256'],
        native_turbines=len(rows),active_turbines=farm.nt,occupied_columns=int(farm.active_columns.size),
        coordinate_words=len(native_columns),coordinate_differences=coordinate_differences,
        parameter_words=farm.params.size,parameter_differences=coefficient_differences,
        numerics='strict' if ENABLED else 'fast',compiler=compile_receipt(),outputs=results,
        production_receipt=farm.receipt())
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture',type=Path);ap.add_argument('--allow-differences',action='store_true')
    args=ap.parse_args();result=replay(args.capture)
    print(json.dumps(result,indent=2))
    assert args.allow_differences or (not result['coordinate_differences'] and not result['parameter_differences']
        and all(v['different']==0 for v in result['outputs'].values())), 'native legacy module replay differs'
