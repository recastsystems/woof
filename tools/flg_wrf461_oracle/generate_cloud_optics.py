"""Record all FLG cloud optical routines from the unmodified WRF object."""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import sys
import numpy as np

ROUTINES=('ice','ice_new_zf','ice_new_comb','ice_new_trop','ice_new_midlat','ice_98','ice_singleice','ice_new_single','water_fl','rain','graup')

def generate(library,path):
    lib=ct.CDLL(str(Path(library).resolve())); n=64; nv=24
    inputs={name:np.empty((n,nv),dtype=np.float32) for name in ('dz','pre','pde','plwc','piwc','prwc','pgwc')}
    with np.load(Path(__file__).resolve().parents[2]/'tests/fixtures/flg/primitives.npz') as primitive:
        inputs['dz'][:]=primitive['dz']
    thresholds=np.array([0.,0.000009999,0.00001,0.000010001,0.001,0.05,0.3,1.],dtype=np.float32)
    radius=np.array([2.,4.18,5.,10.,15.,25.,31.23,45.],dtype=np.float32)
    sizes=np.array([10.,15.,25.,40.,60.,85.,120.,150.],dtype=np.float32)
    for c in range(n):
        for i in range(nv):
            inputs['pre'][c,i]=radius[(c+i)%8]; inputs['pde'][c,i]=sizes[(c+2*i)%8]
            for j,key in enumerate(('plwc','piwc','prwc','pgwc')):
                inputs[key][c,i]=thresholds[(c+i+j)%8]
    outputs={'tau':np.empty((n,11,18,nv),dtype=np.float32),'ssa':np.empty((n,11,18,nv),dtype=np.float32),
             'moments':np.empty((n,11,18,nv,4),dtype=np.float32)}
    def ptr(a): return a.ctypes.data_as(ct.POINTER(ct.c_float))
    def integer(v): return ct.byref(ct.c_int(v))
    for c in range(n):
        for mode,name in enumerate(ROUTINES):
            fn=getattr(lib,'__module_ra_flg_MOD_'+name); fn.restype=None
            for band in range(1,19):
                tau=np.empty(nv,dtype=np.float32); ssa=np.empty_like(tau); moments=np.empty((nv,4),dtype=np.float32,order='F')
                if mode<=8:
                    values=('pre','pde','plwc','piwc','dz') if mode<8 else ('pre','plwc','pde','piwc','dz')
                else: values=('prwc','dz') if mode==9 else ('pgwc','dz')
                fn(integer(nv),integer(nv+1),integer(band),*[ptr(inputs[key][c]) for key in values],ptr(tau),ptr(ssa),ptr(moments))
                outputs['tau'][c,mode,band-1]=tau; outputs['ssa'][c,mode,band-1]=ssa; outputs['moments'][c,mode,band-1]=moments
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**inputs,**outputs)
    receipt={'columns':n,'layers':nv,'routines':list(ROUTINES),'bands':18,
             'wrf_commit':'d66e442fccc04111067e29274c9f9eaccc3cef28',
             'source_sha256':hashlib.sha256(Path('/work/wrf/src/WRF-4.6.1/phys/module_ra_flg.F').read_bytes()).hexdigest(),
             'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'fields':{key:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest(),'finite':bool(np.all(np.isfinite(v)))} for key,v in outputs.items()}}
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not all(row['finite'] for row in receipt['fields'].values()):
        raise ValueError('cloud oracle produced nonfinite words')
    print('cloud fixture bytes',path.stat().st_size)

if __name__=='__main__': generate(sys.argv[1],sys.argv[2])
