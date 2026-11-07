"""Compare the assembled CUDA rad_all with pristine Fortran on a synthetic deck.

This is a component regression check. It is not real WRF-run replay proof.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from woof.core.flg import FLGColumn, SOURCE_SHA256


def inputs():
    n=8; nv=24
    p=np.tile(np.geomspace(15.,1000.,nv+1,dtype=np.float32),(n,1))
    t=np.tile(np.linspace(215.,294.,nv+1,dtype=np.float32),(n,1))
    h=np.tile(np.geomspace(1e-6,.012,nv+1,dtype=np.float32),(n,1))
    o=np.tile(np.linspace(8e-6,1e-7,nv+1,dtype=np.float32),(n,1))
    z=np.zeros((n,nv),dtype=np.float32); cloud=z.copy(); ice=z.copy(); liquid=z.copy()
    for c in range(n):
        if c%4:
            cloud[c,9:16]=[.25,.5,1.][c%4-1]
            ice[c,9:12]=.015
            liquid[c,12:16]=.08
    return dict(pressure=p,temperature=t,humidity=h,ozone=o,ice=ice,ice_size=np.full_like(z,55.),
                liquid=liquid,liquid_radius=np.full_like(z,10.),rain=z.copy(),graupel=z.copy(),
                cloud_fraction=cloud,inhomogeneity=np.ones_like(z),albedo=np.full((n,6),.15,dtype=np.float32),
                emissivity=np.full((n,12),.98,dtype=np.float32),coszen=np.array([.8,.7,.5,.3,-.1,-.3,-.6,-.9],dtype=np.float32),
                solar_constant=np.full(n,1367.,dtype=np.float32),skin_temperature=np.full(n,295.,dtype=np.float32))


def native(library, fields):
    lib=ct.CDLL(str(Path(library).resolve())); n,nv1=fields['pressure'].shape; nv=nv1-1
    ct.c_int.in_dll(lib,'__para_file_MOD_nsubcld').value=(nv-1)//3
    ct.c_int.in_dll(lib,'__para_file_MOD_ngroup').value=3
    ct.c_int.in_dll(lib,'__para_file_MOD_nclouds').value=3*((nv-1)//3)
    for name in ('umo2','umno','umso2','umno2','umch3cl','umco','umcfc11','umcfc12'):
        ct.c_float.in_dll(lib,'__control_para_MOD_'+name).value=0.
    fn=lib.__module_ra_flg_MOD_rad_all; fn.restype=None
    result={name:np.empty((n,nv if name in ('dts','dtir','dt') else nv1),dtype=np.float32) for name in FLGColumn.output_names}
    aerosols=[np.zeros((7,18),dtype=np.float32),np.zeros((7,18),dtype=np.float32),np.zeros((100,18),dtype=np.float32)]
    def integer(v): return ct.byref(ct.c_int(v))
    def scalar(v): return ct.byref(ct.c_float(float(v)))
    def ptr(a): return a.ctypes.data_as(ct.POINTER(ct.c_float))
    for c in range(n):
        ct.c_int.in_dll(lib,'__control_para_MOD_edding').value=0
        ct.c_int.in_dll(lib,'__control_para_MOD_quadra').value=0
        ct.c_int.in_dll(lib,'__control_para_MOD_hemisp').value=0
        args=[integer(nv),integer(nv1),ptr(fields['albedo'][c]),scalar(fields['coszen'][c]),
              scalar(fields['solar_constant'][c]),scalar(fields['skin_temperature'][c]),ptr(fields['emissivity'][c])]
        args.extend(ptr(fields[k][c]) for k in FLGColumn.profile_names+FLGColumn.layer_names)
        args.extend(ptr(a) for a in aerosols)
        args.extend(ptr(result[k][c]) for k in FLGColumn.output_names)
        fn(*args)
    return result


def captured(path):
    rows=[]; results=[]
    with open(path,'rb') as stream:
        while header:=stream.read(8):
            nv,nv1=np.frombuffer(header,dtype='<i4')
            if nv1!=nv+1 or not 4<=nv<=100: raise ValueError('invalid captured vertical extent')
            def read(count):
                data=stream.read(4*int(count))
                if len(data)!=4*count: raise ValueError('truncated native capture')
                return np.frombuffer(data,dtype='<f4').copy()
            row={'albedo':read(6),'coszen':read(1)[0],'solar_constant':read(1)[0],
                 'skin_temperature':read(1)[0],'emissivity':read(12)}
            for key in FLGColumn.profile_names:row[key]=read(nv1)
            for key in FLGColumn.layer_names:row[key]=read(nv)
            outputs={key:read(nv if key in ('dts','dtir','dt') else nv1) for key in FLGColumn.output_names}
            rows.append(row); results.append(outputs)
    if not rows:raise ValueError('native capture is empty')
    return ({key:np.asarray([r[key] for r in rows],dtype=np.float32) for key in rows[0]},
            {key:np.asarray([r[key] for r in results],dtype=np.float32) for key in results[0]})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('library'); parser.add_argument('receipt'); parser.add_argument('--capture'); args=parser.parse_args()
    import cupy as cp
    from woof.core.kernels import module_source
    fields,reference=captured(args.capture) if args.capture else (inputs(),None)
    if reference is None:reference=native(args.library,fields)
    n,nv1=fields['pressure'].shape
    column=FLGColumn(n,nv1-1); got=column(**{k:cp.asarray(v) for k,v in fields.items()}); column.check_errors()
    receipt={'kind':'real native WRF rad_all input replay' if args.capture else 'synthetic component check, not real native forecast replay','source_sha256':SOURCE_SHA256,
             'capture_sha256':hashlib.sha256(Path(args.capture).read_bytes()).hexdigest() if args.capture else None,
             'kernel_sha256':hashlib.sha256(module_source('flg_column').encode()).hexdigest(),'columns':n,'layers':nv1-1,'sunlit_columns':int(np.sum(fields['coszen']>0)),
             'cloudy_sunlit_columns':int(np.sum((fields['coszen']>0)&np.any(fields['cloud_fraction']>0,axis=1))),'fields':{}}
    for name,value in got.items():
        actual=cp.asnumpy(value); expected=reference[name]; misses=np.argwhere(actual.view(np.uint32)!=expected.view(np.uint32))
        receipt['fields'][name]={'words':int(expected.size),'differing_words':len(misses),'first_mismatch':misses[:1].tolist(),
             'finite':bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
             'max_absolute_difference':float(np.max(np.abs(actual-expected))),
             'native_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'cuda_sha256':hashlib.sha256(actual.tobytes()).hexdigest()}
    Path(args.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    if any(r['differing_words'] or not r['finite'] for r in receipt['fields'].values()): raise SystemExit(1)

if __name__=='__main__': main()
