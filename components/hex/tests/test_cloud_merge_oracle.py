"""Production merge against unmodified WRF 4.6.1 Fortran loops."""
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from woof.hex.mpasphys.radshared_v842 import merge_mynn_bl_clouds

@pytest.fixture
def cloud_oracle(tmp_path):
    compiler = shutil.which('gfortran')
    if compiler is None:
        pytest.skip('WRF oracle requires gfortran')
    library = tmp_path / 'cloud.so'
    source = Path(__file__).parent / 'fixtures/cloud_merge_oracle.f90'
    subprocess.run([compiler, '-O0', '-ffp-contract=off', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    return library

def check_oracle(library, xp):
    f=ct.CDLL(str(library)).oracle
    f.argtypes=[ct.c_int,ct.c_int]+[ct.POINTER(ct.c_float)]*6
    rng=np.random.default_rng(276)
    n=4096
    base=[rng.uniform(0,v,n).astype('f4') for v in (2e-6,2e-8,1.,1e-4,1e-5,1.)]
    for arr,threshold in ((base[0],1e-6),(base[1],1e-8),(base[5],.001)):
     t=np.float32(threshold);arr[:4]=[0,np.nextafter(t,np.float32(0)),t,np.nextafter(t,np.float32(1))]
    checks=[]
    for step in (1,2,101):
     native=[x.copy() for x in base];f(n,step,*[x.ctypes.data_as(ct.POINTER(ct.c_float)) for x in native])
     got=[xp.asarray(x).copy() for x in base]
     merge_mynn_bl_clouds(*got[:3],qc_bl=got[3],qi_bl=got[4],cldfra_bl=got[5],itimestep=step,xp=xp)
     for k in range(6):
      a=xp.asnumpy(got[k]) if xp is not np else got[k]
      np.testing.assert_array_equal(a,native[k])
     # Demonstrate sensitivity: original patch must fail this same oracle.
     old=[base[0]+base[3],base[1]+base[4],np.maximum(base[2],base[5])]
     bad=[int(np.count_nonzero(a!=b)) for a,b in zip(old,native[:3])]
     assert all(x>0 for x in bad)
     checks.append({'itimestep':step,'words_checked':6*n,'old_patch_mismatches':bad})
    print(json.dumps({'passed':True,'backend':xp.__name__,'checks':checks},indent=2))

def test_cloud_merge_cpu(cloud_oracle):
    check_oracle(cloud_oracle, np)

@pytest.mark.gpu
def test_cloud_merge_gpu(cloud_oracle):
    import cupy
    check_oracle(cloud_oracle, cupy)
