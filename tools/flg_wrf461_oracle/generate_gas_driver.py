"""Record both complete WRF FLG gas arms on every quadrature point."""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import sys
import numpy as np
from woof.core.flg import gas_points,load_tables
from generate_gas_kernel import FLOAT_GLOBALS,INT_GLOBALS

def generate(library,path):
    lib=ct.CDLL(str(Path(library).resolve())); n=64; nv1=25; nv=24
    with np.load(Path(__file__).resolve().parents[2]/'tests/fixtures/flg/gas_helpers.npz') as f:
        pp=f['pressure'].copy(); pt=f['temperature'].copy(); ph=f['humidity'].copy()
    # Full gas-arm calibration uses the primary column oracle's atmospheric range.
    # Temperature clamp extrapolation is graded separately in gas_helpers.npz.
    for c in range(n):
        x=np.linspace(0.,1.,nv1,dtype=np.float32)
        pp[c]=10.+990.*x*x
        pt[c]=210.+(84.+c%7)*x
    po=np.empty_like(pp)
    for c in range(n): po[c]=np.linspace(0.000003,0.00000001,nv1,dtype=np.float32)
    rows=gas_points(); count=len(rows)
    values=np.empty((n,11),dtype=np.float32); switches=np.empty((n,20),dtype=np.int32)
    cells={key:ct.c_float.in_dll(lib,'__control_para_MOD_'+key) for key in FLOAT_GLOBALS}
    flags={key:ct.c_int.in_dll(lib,'__control_para_MOD_'+key) for key in INT_GLOBALS}
    baseline=np.array([cells[key].value for key in FLOAT_GLOBALS],dtype=np.float32)
    result={'optical_depth':np.empty((n,count,nv),dtype=np.float32),'weights':np.empty((n,count),dtype=np.float32),
            'temperature_out':np.empty((n,count,nv1),dtype=np.float32)}
    functions={0:lib.__module_ra_flg_MOD_gases,1:lib.__module_ra_flg_MOD_gases_new}
    for fn in functions.values(): fn.restype=None
    def ptr(a): return a.ctypes.data_as(ct.POINTER(ct.c_float))
    def integer(v): return ct.byref(ct.c_int(int(v)))
    for c in range(n):
        values[c]=baseline*np.float32(1.-0.05*(c%5))
        switches[c]=1
        if c%4==1: switches[c,13]=0
        if c%4==2: switches[c,[12,14,15,16,17,18,19]]=0
        for j,key in enumerate(FLOAT_GLOBALS): cells[key].value=float(values[c,j])
        for j,key in enumerate(INT_GLOBALS): flags[key].value=int(switches[c,j])
        for g,(arm,band,one,two) in enumerate(rows):
            p=pp[c].copy(); t=pt[c].copy(); h=ph[c].copy(); o=po[c].copy(); hk=ct.c_float()
            args=[integer(nv),integer(nv1),integer(band),integer(one)]
            if arm==1: args.append(integer(two))
            args.extend([ct.byref(hk),ptr(p),ptr(t),ptr(h),ptr(o),ptr(result['optical_depth'][c,g])])
            functions[int(arm)](*args)
            result['weights'][c,g]=hk.value; result['temperature_out'][c,g]=t
    path=Path(path); np.savez_compressed(path,pressure=pp,temperature=pt,humidity=ph,ozone=po,gas_values=values,switches=switches,points=rows,**result)
    receipt={'columns':n,'levels':nv1,'points':count,'old_points':int(np.count_nonzero(rows[:,0]==0)),
             'new_points':int(np.count_nonzero(rows[:,0]==1)),'gas_values_order':list(FLOAT_GLOBALS),'switch_order':list(INT_GLOBALS),
             'wrf_commit':'d66e442fccc04111067e29274c9f9eaccc3cef28',
             'source_sha256':hashlib.sha256(Path('/work/wrf/src/WRF-4.6.1/phys/module_ra_flg.F').read_bytes()).hexdigest(),
             'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'fields':{key:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest(),'finite':bool(np.all(np.isfinite(v)))} for key,v in result.items()}}
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not all(row['finite'] for row in receipt['fields'].values()): raise ValueError('nonfinite gas driver reference')
    print('gas driver fixture',count,'points',path.stat().st_size,'bytes')

if __name__=='__main__': generate(sys.argv[1],sys.argv[2])
