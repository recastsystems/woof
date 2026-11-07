"""Stock WRF 4.6.1 word comparison of the GFDL initialization primitives.

Run only on an authorized card under the card queue. Both arithmetic modes
use the same kernel and the same FTZ/contraction-preserving loader.
"""
from pathlib import Path
import csv
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('GPUWM_NO_LOCAL_GPU') == '1', reason='CPU-only lane test run')
FIX=Path(__file__).parent/'fixtures/gfdleta'

def _words(filename):
    with (FIX/filename).open() as stream:
        return {(r['field'],int(r['column']),int(r['index'])):int(r['bits'],16)
                for r in csv.DictReader(stream)}

def _kernel(name):
    from woof.core.kernels import get_kernel
    return get_kernel('gfdleta_init',name)

def _same(actual,want):
    import cupy as cp
    words=cp.asnumpy(actual).view(np.uint32)
    assert np.array_equal(words,np.asarray(want,dtype=np.uint32)), (
        'float32 word mismatch', np.flatnonzero(words != want)[:10].tolist())

def test_gausin_axsd_all_310_words():
    import cupy as cp
    inp=_words('primitives-inputs.csv');ref=_words('primitives-expected.csv')
    x=np.array([inp[('GAUSIN',c,0)] for c in range(1,311)],np.uint32).view(np.float32)
    y=cp.empty(310,cp.float32)
    _kernel('gd_gaussian')((10,),(32,),(cp.asarray(x),y,np.int32(310)))
    _same(y,[ref[('GAUSIN',c,0)] for c in range(1,311)])

def test_solard_64_dates():
    import cupy as cp
    inp=_words('primitives-inputs.csv');ref=_words('primitives-expected.csv')
    x=np.array([[inp[(f,c,0)] for f in ('YEAR','MONTH','DAY','HOUR')] for c in range(1,65)],np.int32)
    y=cp.empty(64,cp.float32)
    _kernel('gd_solar')((2,),(32,),(cp.asarray(x),y,np.int32(64)))
    _same(y,[ref[('SOLARD',c,0)] for c in range(1,65)])

def test_clo89_64_columns():
    import cupy as cp
    inp=_words('primitives-inputs.csv');ref=_words('primitives-expected.csv')
    n=12;nv=n+1
    counts=np.array([inp[('NCLDS',c,0)] for c in range(1,65)],np.int32)
    ca=np.array([[inp[('CAMT',c,k)] for k in range(1,nv+1)] for c in range(1,65)],np.uint32).view(np.float32)
    kb=np.array([[inp[('KBTM',c,k)] for k in range(1,nv+1)] for c in range(1,65)],np.int32)
    kt=np.array([[inp[('KTOP',c,k)] for k in range(1,nv+1)] for c in range(1,65)],np.int32)
    y=cp.empty(64*nv*nv,cp.float32)
    _kernel('gd_clo89')((2,),(32,),(cp.asarray(ca),cp.asarray(counts),cp.asarray(kb),cp.asarray(kt),y,np.int32(64),np.int32(n)))
    _same(y,[ref[('CLO89',c,k)] for c in range(1,65) for k in range(1,nv*nv+1)])

def test_no_ftz_and_no_contraction_on_device():
    import cupy as cp
    x=np.array([1.e-30,1.e-10,1.e-39,2.,1.0000001192092896,1.0000001192092896,-1.000000238418579],np.float32)
    with np.errstate(under='ignore'):
        want=np.array([x[0]*x[1],x[2]/x[3],np.float32(x[4]*x[5])+x[6]],np.float32)
    y=cp.empty(3,cp.float32)
    _kernel('gd_fp_probe')((1,),(1,),(cp.asarray(x),y))
    _same(y,want.view(np.uint32))
    assert want[0]>0 and want[1]>0 and want[2]==0
