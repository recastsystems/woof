"""Record unmodified WRF primitives through their compiled Fortran entry points."""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import sys
import numpy as np

def generate(library,out):
    lib=ct.CDLL(str(Path(library).resolve()))
    functions={name:getattr(lib,'__module_ra_flg_MOD_'+name) for name in ('thicks','rayle2','rayle','planck','o3prof')}
    for fn in functions.values(): fn.restype=None
    n=64; nv1=25; nv=nv1-1
    shape={'dz':(n,nv),'trp':(n,nv),'tr':(n,18,nv),'wr':(n,18,nv),'wwr':(n,18,nv,4),
           'bf':(n,12,nv1),'bs':(n,12),'ozone':(n,5,nv1)}
    outputs={key:np.empty(value,dtype=np.float32) for key,value in shape.items()}
    pp=np.empty((n,nv1),dtype=np.float32); pt=np.empty_like(pp)
    u0=np.linspace(-1.,1.,n,dtype=np.float32); pts=np.linspace(180.,320.,n,dtype=np.float32)
    for c in range(n):
        pp[c]=np.geomspace(0.0001+(c%8)*0.01,1050.-(c%7)*20.,nv1).astype(np.float32)
        pt[c]=np.linspace(190.+(c%8),280.+(c%11),nv1,dtype=np.float32)
        p=pp[c].copy(); t=pt[c].copy(); ph=np.zeros(nv1,dtype=np.float32); po=ph.copy()
        n1=ct.c_int(nv1); nl=ct.c_int(nv)
        def ptr(array): return array.ctypes.data_as(ct.POINTER(ct.c_float))
        def integer(value): return ct.byref(ct.c_int(value))
        for field,name in [('dz','thicks'),('trp','rayle2')]:
            value=np.empty(nv,dtype=np.float32)
            functions[name](ct.byref(nl),ct.byref(n1),ptr(p),ptr(t),ptr(ph),ptr(po),ptr(value))
            outputs[field][c]=value
        for band in range(1,19):
            tr=np.empty(nv,dtype=np.float32); wr=np.empty_like(tr); wwr=np.empty((nv,4),dtype=np.float32,order='F')
            functions['rayle'](ct.byref(nl),ct.byref(n1),integer(band),ptr(outputs['trp'][c]),ptr(tr),ptr(wr),ptr(wwr),ct.byref(ct.c_float(u0[c])))
            outputs['tr'][c,band-1]=tr; outputs['wr'][c,band-1]=wr; outputs['wwr'][c,band-1]=wwr
        for band in range(7,19):
            bf=np.empty(nv1,dtype=np.float32); bs=ct.c_float()
            functions['planck'](ct.byref(nl),ct.byref(n1),integer(band),ct.byref(ct.c_float(pts[c])),ptr(p),ptr(t),ptr(ph),ptr(po),ptr(bf),ct.byref(bs))
            outputs['bf'][c,band-7]=bf; outputs['bs'][c,band-7]=bs.value
        for profile in range(1,6):
            reverse=p[::-1].copy(); ozone=np.empty_like(reverse)
            functions['o3prof'](integer(profile),integer(1),ct.byref(n1),ptr(reverse),ptr(ozone))
            outputs['ozone'][c,profile-1]=ozone[::-1]
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,pressure=pp,temperature=pt,coszen=u0,surface_temperature=pts,**outputs)
    receipt={'columns':n,'levels':nv1,'routine_sources':{'thicks':9561,'rayle2':10869,'rayle':10888,'planck':12786,'o3prof':15275},
             'compiler_flags':'-O0 -fPIC -ffree-form -ffree-line-length-none -fno-tree-vectorize -ffp-contract=off',
             'wrf_commit':'d66e442fccc04111067e29274c9f9eaccc3cef28',
             'fixture_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
             'source_sha256':hashlib.sha256(Path('/work/wrf/src/WRF-4.6.1/phys/module_ra_flg.F').read_bytes()).hexdigest(),
             'fields':{key:{'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()} for key,value in outputs.items()}}
    out.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__': generate(sys.argv[1],sys.argv[2])
