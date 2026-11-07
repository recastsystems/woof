"""WRF 4.6.1 word comparison of staged GFDL LW table interpolations."""
from pathlib import Path
import csv
import os
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('GPUWM_NO_LOCAL_GPU')=='1',reason='CPU-only lane test run')
FIX=Path(__file__).parent/'fixtures/gfdleta'

def _words(filename):
    with (FIX/filename).open() as stream:
        return {(r['field'],int(r['column']),int(r['index'])):int(r['bits'],16) for r in csv.DictReader(stream)}

def _f(inp,field,nc,nk):
    return np.array([inp[(field,c,k)] for c in range(1,nc+1) for k in range(1,nk+1)],np.uint32).view(np.float32)

def _k(name):
    from woof.core.kernels import get_kernel
    return get_kernel('gfdleta_lw_helpers',name)

def _compare(got,ref,field,nc,keys):
    import cupy as cp
    words=cp.asnumpy(got).view(np.uint32).reshape(nc,-1)
    actual=np.array([words[c-1,k-1] for c in range(1,nc+1) for k in keys])
    expected=np.array([ref[(field,c,k)] for c in range(1,nc+1) for k in keys],np.uint32)
    bad=np.flatnonzero(actual!=expected)
    assert bad.size==0,(field,int(bad.size),[(int(i),int(actual[i]),int(expected[i])) for i in bad[:10]])

def test_e3v88_1600_words():
    import cupy as cp
    inp=_words('lw-inputs.csv');ref=_words('lw-expected.csv')
    tv=cp.asarray(_f(inp,'TV',64,25));av=cp.asarray(_f(inp,'AV',64,25))
    table=cp.asarray(_f(inp,'EM3',1,5040));out=cp.empty(1600,cp.float32)
    _k('gd_e3v88')((50,),(32,),(tv,av,table,out,np.int32(1600)))
    _compare(out,ref,'E3V88',64,range(1,26))

def test_e2spec_128_words():
    import cupy as cp
    inp=_words('lw-inputs.csv');ref=_words('lw-expected.csv')
    args=[cp.asarray(_f(inp,f,64,13)) for f in ('PHI','FX','DT')]
    args += [cp.asarray(_f(inp,f,1,5040)) for f in ('T1','T2','T4')]
    out=cp.zeros(64*13,cp.float32)
    _k('gd_e2spec')((2,),(32,),(*args,out,np.int32(64),np.int32(12)))
    _compare(out,ref,'E2SPEC',64,range(1,3))

def test_e290_704_and_640_words():
    import cupy as cp
    inp=_words('lw-inputs.csv');ref=_words('lw-expected.csv')
    args=[cp.asarray(_f(inp,f,64,13)) for f in ('PHI','FX','DT')]
    args += [cp.asarray(_f(inp,f,1,5040)) for f in ('T1','T2','T4')]
    em=cp.zeros(64*13,cp.float32);eb=cp.zeros_like(em)
    _k('gd_e290')((2,),(32,),(*args,em,eb,np.int32(64),np.int32(12),np.int32(3)))
    _compare(em,ref,'E290',64,range(3,14))
    _compare(eb,ref,'E290B',64,range(3,13))

def test_e1e290_4864_words():
    import cupy as cp
    inp=_words('lw-inputs.csv');ref=_words('lw-expected.csv')
    args=[cp.asarray(_f(inp,f,64,13)) for f in ('PHI','FX1','DT1','FX','DT')]
    args += [cp.asarray(_f(inp,f,1,5040)) for f in ('T1','T2','T4','EM1','EM1VW')]
    em=cp.empty(64*13,cp.float32);g1=cp.empty_like(em);g3=cp.empty_like(em);g4=cp.empty_like(em)
    g2=cp.empty(64*12,cp.float32);g5=cp.empty_like(g2)
    _k('gd_e1e290')((2,),(32,),(*args,em,g1,g2,g3,g4,g5,np.int32(64),np.int32(12)))
    for field,a in [('E1EMISS',em),('G1',g1),('G3',g3),('G4',g4)]:
        _compare(a,ref,field,64,range(1,14))
    for field,a in [('G2',g2),('G5',g5)]:
        _compare(a,ref,field,64,range(1,13))
