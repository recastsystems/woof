"""Record four callable WRF FLG stream solvers on 64 optical profiles."""
from pathlib import Path
import ctypes as ct
import hashlib,json,sys
import numpy as np

def generate(library,path):
    lib=ct.CDLL(str(Path(library).resolve())); n=64; nv=24; nv1=25
    bands=np.column_stack((1+np.arange(n)%6,7+np.arange(n)%12)).astype(np.int32)
    alb=np.linspace(0.,0.9,n,dtype=np.float32); cosine=np.linspace(.02,.99,n,dtype=np.float32)
    f0=np.full(n,np.float32(1.)/np.float32(3.14159),dtype=np.float32)
    emiss=np.linspace(.8,1.,n,dtype=np.float32)
    with np.load(Path(__file__).resolve().parents[2]/'tests/fixtures/flg/primitives.npz') as p:
        bf=np.array([p['bf'][c,bands[c,1]-7] for c in range(n)],dtype=np.float32)
        bs=np.array([p['bs'][c,bands[c,1]-7] for c in range(n)],dtype=np.float32)
    phase=np.empty((n,4,nv),dtype=np.float32); scattering=np.empty((n,nv),dtype=np.float32); depth=np.empty_like(scattering)
    for c in range(n):
        g=np.linspace(.05,.75,nv,dtype=np.float32)
        for k in range(4): phase[c,k]=np.float32(3+2*k)*(g**(k+1))
        scattering[c]=np.linspace(0.,.95,nv,dtype=np.float32)
        if c%4==0: scattering[c]=0.; phase[c]=0.
        if c%4==1: phase[c,0]=0.; phase[c,1]=.5; phase[c,2:]=0.
        thickness=np.linspace(.001,.03+(c%8)*.01,nv,dtype=np.float32)
        running=np.float32(0.)
        for k in range(nv): running=np.float32(running+thickness[k]); depth[c,k]=running
    up=np.empty((n,4,nv1),dtype=np.float32); down=np.empty_like(up)
    direct=np.empty((n,nv1),dtype=np.float32); diffuse=np.empty_like(direct)
    names=('qfts','qftsts','qfti','qftisf'); controls=('d4s','d2s','d4ir','d2ir','edding','quadra','hemisp')
    def integer(v): return ct.byref(ct.c_int(int(v)))
    def scalar(v): return ct.byref(ct.c_float(float(v)))
    def ptr(a): return a.ctypes.data_as(ct.POINTER(ct.c_float))
    for c in range(n):
        for arm,name in enumerate(names):
            values=(arm!=1,arm==1,arm==2,arm!=2,arm==1,False,arm!=1)
            for key,value in zip(controls,values): ct.c_int.in_dll(lib,'__control_para_MOD_'+key).value=int(value)
            fn=getattr(lib,'__module_ra_flg_MOD_'+name); fn.restype=None
            args=[integer(nv),integer(nv1),integer(bands[c,0] if arm<2 else bands[c,1])]
            if arm<2: args.extend([scalar(alb[c]),scalar(cosine[c]),scalar(f0[c])])
            else: args.extend([scalar(emiss[c]),ptr(bf[c]),scalar(bs[c])])
            args.extend([ptr(phase[c,k]) for k in range(4)])
            args.extend([ptr(scattering[c]),ptr(depth[c]),ptr(up[c,arm]),ptr(down[c,arm])])
            if arm==0: args.extend([ptr(direct[c]),ptr(diffuse[c])])
            fn(*args)
    result={'up':up,'down':down,'direct':direct,'diffuse':diffuse}
    path=Path(path); np.savez_compressed(path,bands=bands,albedo=alb,coszen=cosine,f0=f0,emissivity=emiss,bf=bf,bs=bs,moments=phase,scattering=scattering,depth=depth,**result)
    receipt={'columns':n,'layers':nv,'routines':list(names),'wrf_commit':'d66e442fccc04111067e29274c9f9eaccc3cef28',
             'source_sha256':hashlib.sha256(Path('/work/wrf/src/WRF-4.6.1/phys/module_ra_flg.F').read_bytes()).hexdigest(),
             'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'fields':{key:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest(),'finite':bool(np.all(np.isfinite(v)))} for key,v in result.items()}}
    path.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not all(row['finite'] for row in receipt['fields'].values()): raise ValueError('nonfinite stream reference')
    print('stream fixture',path.stat().st_size,'bytes')

if __name__=='__main__': generate(sys.argv[1],sys.argv[2])
