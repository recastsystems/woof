"""Grade all outputs against the independently compiled WRF driver."""
import argparse, ctypes, json, hashlib
from pathlib import Path
import numpy as np
from woof.core.fp32_ulp import monotone_fp32_key

def differences(a,b):
    aa=monotone_fp32_key(a); bb=monotone_fp32_key(b)
    return {'max_ulp':int(np.max(np.abs(aa-bb))), 'different_words':int(np.count_nonzero(a.view('u4')!=b.view('u4'))), 'max_abs':float(np.max(np.abs(a-b)))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cpu');ap.add_argument('--out',required=True);args=ap.parse_args()
    data=np.load(Path(__file__).with_name('columns.npz')); result={}; worst=0
    if args.cpu:
        fun=ctypes.CDLL(args.cpu).gsfcsw_columns_kernel
        ptr=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS')
        fun.argtypes=[ptr,ptr,ptr,ptr]+[ctypes.c_int]*6+[ctypes.c_float]*2
        tables=np.fromfile(Path('woof/data/wrf_radiation/gsfcsw/coefficients.f32'),dtype='<f4')
    else:
        import cupy as cp
        from woof.core.gsfcsw import packed_columns
    for nlay in (8,48):
        n=nlay+1; actual=[]
        for col in range(64):
            inp=data[f'in_{nlay}'][col].copy(); warm,fqi,fqn=map(int,data[f'flags_{nlay}'][col]); day,solar,latitude=map(float,data[f'meta_{nlay}'][col])
            if args.cpu:
                out=np.zeros(3*n+2,'f4'); fun(tables,inp,out,np.zeros(32*32677,'f4'),1,nlay,warm,fqi,fqn,int(day),solar,latitude)
            else:
                out=cp.asnumpy(packed_columns(cp.asarray(inp[None,:]),nlay,warm_rain=bool(warm),f_qi=bool(fqi),f_qndrop=bool(fqn),julday=int(day),solcon=solar,center_lat=latitude)[0])
            actual.append(out)
        actual=np.asarray(actual); wanted=data[f'out_{nlay}']; fields={}
        for name,region in [('RTHRATEN',slice(0,nlay)),('TAUCLDI',slice(n,2*n-1)),('TAUCLDC',slice(2*n,3*n-1)),('GSW',slice(3*n,3*n+1)),('RSWTOA',slice(3*n+1,3*n+2))]:
            fields[name]=differences(actual[:,region],wanted[:,region]); fields[name]['actual_sha256']=hashlib.sha256(actual[:,region].tobytes()).hexdigest(); fields[name]['expected_sha256']=hashlib.sha256(wanted[:,region].tobytes()).hexdigest(); worst=max(worst,fields[name]['max_ulp'])
        result[str(nlay)]={'columns':64,'fields':fields}
    result['max_ulp']=worst
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
    if worst or any(f['different_words'] for k,v in result.items() if k!='max_ulp' for f in v['fields'].values()): raise SystemExit(1)
if __name__=='__main__': main()
