"""Full GFDL TABLE output words against stock WRF 4.6.1."""
from pathlib import Path
import csv
import os
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('GPUWM_NO_LOCAL_GPU')=='1',reason='CPU-only lane test run')

def test_all_table_words():
    import cupy as cp
    from woof.core.kernels import get_kernel
    names=('EM1','TABLE1','EM3','EM1WDE','SOURCE','TABLE2','TABLE3','DSRCE')
    a={f:cp.empty(420 if f in ('SOURCE','DSRCE') else 5040,cp.float32) for f in names}
    aux=cp.empty(4,cp.float32)
    get_kernel('gfdleta_tables','gd_table_rows')((1,),(32,),(*(a[f] for f in names[:5]),aux))
    get_kernel('gfdleta_tables','gd_table_derivatives')((158,),(32,),(a['TABLE1'],a['SOURCE'],a['TABLE2'],a['TABLE3'],a['DSRCE']))
    ref={}
    p=Path(__file__).parent/'fixtures/gfdleta/table-expected.csv'
    with p.open() as stream:
        for row in csv.DictReader(stream):ref.setdefault(row['field'],[]).append(int(row['bits'],16))
    for name in names:
        got=cp.asnumpy(a[name]).view(np.uint32)
        want=np.array(ref[name],np.uint32)
        bad=np.flatnonzero(got!=want)
        assert bad.size==0,(name,int(bad.size),[(int(i),int(got[i]),int(want[i])) for i in bad[:10]])

    got=cp.asnumpy(aux).view(np.uint32)
    assert np.array_equal(got,np.array(ref['TABLE_AUX'],np.uint32))
