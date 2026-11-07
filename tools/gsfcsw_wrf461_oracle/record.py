"""Record small independent column fixtures, including all driver outputs."""
import ctypes, hashlib, json, sys
from pathlib import Path
import numpy as np

def fixture(nlay,col):
    n=nlay+1
    z=np.arange(n,dtype=np.float32)/np.float32(nlay)
    p8=np.float32(100000)-np.float32(30000 if nlay==8 else 95000)*z
    p=(p8+np.r_[p8[1:],np.float32(2000)])/np.float32(2)
    t=np.float32(296)-np.float32(35 if nlay==8 else 85)*z
    ex=(p/np.float32(100000))**np.float32(287/1004.5)
    qv=np.float32(0.016)*(np.float32(1)-z)**np.float32(3)
    qc=np.zeros(n,'f4'); qi=qc.copy(); cf=qc.copy(); qn=qc.copy()
    mode=col%8
    for k in range(nlay):
        if 0.15<z[k]<0.8 and mode in (1,2,3,4,5,6):
            qc[k]=0.003 if mode in (1,3,4,5,6) else 0
            qi[k]=0.002 if mode in (2,3) else 0
            cf[k]=1 if mode in (1,2) else 0.45
            qn[k]=1e8 if mode==5 else 0
    mu=np.float32([0.8,0.03,0.99,-0.1,0.,1.e-10,0.45,0.15][(col//8)%8])
    alb=np.float32(0.85 if mode==7 else 0.15)
    inp=np.concatenate([t,p,p8,ex,qv,qc,qi,cf,qn,[mu,alb]]).astype('f4')
    warm=int(mode==4); fqi=int(mode not in (4,6)); fqn=int(mode==5)
    return inp,warm,fqi,fqn

def main():
    build=Path(sys.argv[1]); lib=ctypes.CDLL(str(build/'oracle.so')); fun=lib.gsfc_column
    ptr=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS')
    fun.argtypes=[ctypes.c_int]*5+[ctypes.c_float]*2+[ptr,ptr]
    dest=Path(__file__).parent/'columns.npz'
    previous=np.load(dest) if dest.is_file() else None
    arrays={}; summaries=[]
    for nlay in (8,48):
        inp=[]; out=[]; flags=[]; metadata=[]
        for col in range(64):
            a,w,qi,qn=fixture(nlay,col); b=np.zeros(3*(nlay+1)+2,'f4')
            latitude=[-75.,-45.,-15.,0.,15.,45.,60.,75.][col%8];day=[1,80,81,172,265,266,355,366][col//8];solar=np.float32(1310+14*(col//8))
            if previous is not None:
                # Rebuilding an oracle changes only native output words.
                # NumPy/libm upgrades must not change the archived inputs.
                a=previous[f'in_{nlay}'][col].copy()
                w,qi,qn=map(int,previous[f'flags_{nlay}'][col])
                day,solar,latitude=previous[f'meta_{nlay}'][col]
                day=int(day)
            fun(nlay,w,qi,qn,day,solar,latitude,a,b)
            metadata.append([day,float(solar),latitude])
            if not np.all(np.isfinite(b)): raise RuntimeError(f'nonfinite WRF column {col}')
            inp.append(a); out.append(b); flags.append([w,qi,qn])
        arrays[f'in_{nlay}']=np.asarray(inp); arrays[f'out_{nlay}']=np.asarray(out); arrays[f'flags_{nlay}']=np.asarray(flags,'i4');arrays[f'meta_{nlay}']=np.asarray(metadata,'f4')
        summaries.append({'layers':nlay,'columns':64,'sha256':hashlib.sha256(arrays[f'out_{nlay}'].tobytes()).hexdigest()})
    if previous is not None: previous.close()
    np.savez(dest,**arrays)
    (build/'receipt.json').write_text(json.dumps({
        'wrf_commit':(build/'wrf-commit.txt').read_text().strip(),
        'module_sha256':'da257cf060c385c3aa4d6c3a737ec8c10773bd52034786d954bab0a0dc290766',
        'compiler':(build/'compiler.txt').read_text().strip(),
        'libc':(build/'libc.txt').read_text().strip(),
        'flags':['-O0','-ffp-contract=off','-fno-tree-vectorize'],
        'oracle_sha256':hashlib.sha256((build/'oracle.so').read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'fixtures':summaries,'fields':['RTHRATEN','TAUCLDI','TAUCLDC','GSW','RSWTOA']},indent=2)+'\n')
    print((build/'receipt.json').read_text())
if __name__=='__main__': main()
