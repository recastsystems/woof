"""Replay each native stress call, including its unchanged words."""
from pathlib import Path
import sys,json,os
import numpy as np

def main():
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.verify.advect_oracle import measure_words
    records={}
    for path in sorted(Path(sys.argv[1]).glob('stress-*.bin')):
        with path.open('rb') as f:
            h=np.fromfile(f,dtype='<i4',count=14)
            version,mode,ims,ime,jms,jme,kms,kme,i0,i1,j0,j1,k0,k1=map(int,h)
            shape=(jme-jms+1,kme-kms+1,ime-ims+1)
            def a3():return np.fromfile(f,dtype='<f4',count=int(np.prod(shape))).reshape(shape).transpose(1,0,2).copy()
            km,rho,deform,old=(a3() for _ in range(4))
            fnm,fnp=(np.fromfile(f,dtype='<f4',count=shape[1]) for _ in range(2))
            reference=a3(); assert not f.read()
        arrays=[cp.asarray(a) for a in (km,rho,deform,fnm,fnp,old)]
        nz,ny,nx=km.shape
        # 13/23's native loop leaves the bottom interface unchanged.
        if mode in (2,3):k0+=1
        get_kernel('sfs_output','wrf_stress_output_raw')(
            ((nx+127)//128,ny,nz),(128,1,1),tuple(arrays+list(map(np.int32,
            (nz,ny,nx,mode,i0-ims,i1-ims,j0-jms,j1-jms,k0-kms,k1-kms)))))
        got=cp.asnumpy(arrays[-1]); records[path.name]=measure_words(got,reference)
        print(path.name,json.dumps(records[path.name]),flush=True)
    Path(sys.argv[2]).write_text(json.dumps(dict(strict=os.getenv('GPUWM_WRF_EXACT'),rows=records),indent=2)+'\n')
if __name__=='__main__':main()
