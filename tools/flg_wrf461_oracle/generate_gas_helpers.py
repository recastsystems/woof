"""Record FLG interpolation and water-vapor continuum from unmodified WRF."""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import sys
import numpy as np
from woof.core.flg import load_tables

def generate(library,path):
    lib=ct.CDLL(str(Path(library).resolve())); n=64; nv1=25; nv=24
    tables=load_tables()
    with np.load(Path(__file__).resolve().parents[2]/'tests/fixtures/flg/primitives.npz') as primitive:
        pressure=primitive['pressure'].copy(); temperature=primitive['temperature'].copy()
    temperature[::4,0]=170.; temperature[1::4,-1]=350.
    humidity=np.empty_like(pressure)
    for c in range(n): humidity[c]=np.linspace(1e-10,0.018+(c%4)*0.002,nv1,dtype=np.float32)
    results={'pressure':pressure,'temperature':temperature,'humidity':humidity}; sources={}
    def ptr(a): return a.ctypes.data_as(ct.POINTER(ct.c_float))
    def integer(v): return ct.byref(ct.c_int(v))
    for method,routine,nodes in [('shortwave','qks',11),('longwave','qki',19),('ozone','qkio3',19)]:
        candidates=[(name,value) for name,value in sorted(tables.items()) if value.ndim>=2 and value.shape[:2]==(3,nodes)]
        coeff=np.empty((n,nodes,3),dtype=np.float32); fkg=np.empty_like(pressure); tout=np.empty_like(temperature)
        fn=getattr(lib,'__module_ra_flg_MOD_'+routine); fn.restype=None; selected=[]
        for c in range(n):
            name,value=candidates[c%len(candidates)]
            slice_value=value if value.ndim==2 else value[:,:,c%value.shape[2]]
            matrix=np.asfortranarray(slice_value,dtype=np.float32); coeff[c]=matrix.T
            p=pressure[c].copy(); t=temperature[c].copy(); ph=humidity[c].copy(); po=np.zeros(nv1,dtype=np.float32)
            fn(integer(nv),integer(nv1),ptr(matrix),ptr(fkg[c]),ptr(p),ptr(t),ptr(ph),ptr(po))
            tout[c]=t; selected.append({'table':name,'gpoint':None if value.ndim==2 else c%value.shape[2]})
        results[method+'_coefficients']=coeff; results[method+'_fkg']=fkg; results[method+'_temperature_out']=tout
        sources[method]=selected
    continuum=np.empty((n,18,nv),dtype=np.float32)
    fn=lib.__module_ra_flg_MOD_gascon; fn.restype=None
    for c in range(n):
        for band in range(1,19):
            p=pressure[c].copy(); t=temperature[c].copy(); h=humidity[c].copy(); po=np.zeros(nv1,dtype=np.float32)
            fn(integer(nv),integer(nv1),integer(band),ptr(continuum[c,band-1]),ptr(p),ptr(t),ptr(h),ptr(po))
    results['continuum']=continuum
    path=Path(path); np.savez_compressed(path,**results)
    receipt={'columns':n,'levels':nv1,'wrf_commit':'d66e442fccc04111067e29274c9f9eaccc3cef28',
        'source_sha256':hashlib.sha256(Path('/work/wrf/src/WRF-4.6.1/phys/module_ra_flg.F').read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'coefficient_slices':sources,
        'fields':{key:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for key,v in results.items() if not key.endswith('_coefficients') and key not in ('pressure','temperature','humidity')}}
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not all(np.all(np.isfinite(v)) for v in results.values()): raise ValueError('nonfinite gas helper fixture')

if __name__=='__main__': generate(sys.argv[1],sys.argv[2])
