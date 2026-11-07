"""Full ETARA LW/SW column outputs against byte-unmodified WRF 4.6.1.

The fixture tables isolate the new column core from initialization. Run
initialization acceptance separately before admitting the forecast scheme.
"""
from pathlib import Path
import os
import json
import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('GPUWM_NO_LOCAL_GPU')=='1',reason='CPU-only check')
FIX=Path(__file__).parent/'fixtures/gfdleta/column'

@pytest.mark.parametrize('n',[12,24,42,60])
def test_full_column_core(n):
    import cupy as cp
    from woof.core.kernels import get_kernel
    with np.load(FIX/f'columns-{n}.npz') as fixture:
        f=cp.asarray(fixture['floats']);i=cp.asarray(fixture['integers']);table=cp.asarray(fixture['table'])
        layout=json.loads((Path(__file__).parents[1]/'woof/core/gfdleta_column_layout.json').read_text())
        work=cp.empty((8,layout['scratch_bytes']),cp.uint8)
        refs=fixture['reference'];indices=fixture['indices'];columns=fixture['columns'];fields=fixture['fields']
        get_kernel('gfdleta_column','gd_radiation_columns')((1,),(32,),(table,f,i,work,np.int32(8)))
        words=cp.asnumpy(f).view(np.uint32)
        got=words[columns,indices]
        bad=np.flatnonzero(got!=refs)
        assert not len(bad),[(str(fields[k]),int(columns[k])+1,int(indices[k]),f'{got[k]:08X}',f'{refs[k]:08X}') for k in bad[:12]]
