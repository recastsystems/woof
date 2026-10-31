import json
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.mpasphys.pool_v842 import NativePoolHydration


@pytest.mark.parametrize('kind,shape', [('real', [2,3]),('integer', []),('real', [0,3])])
def test_validated_zero_transport_allocates_native_storage_without_temporary(tmp_path, kind, shape):
    row=dict(pool='state',name='zero',pool_name='zero',kind=kind,dimensions=[],shape=shape,nlevels=2)
    record={k:row[k] for k in ('pool','name','kind','dimensions','shape')}
    record['fill']=0
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema='mpas-native-pool-payload-v1',fields=[record])))
    seen=[]
    native=SimpleNamespace(pool=lambda name:None,
        put_zero_field=lambda *args,**kw:seen.append((args,kw)))
    hydration=NativePoolHydration(native,SimpleNamespace(fields=[row],configs={},dimensions={},indices={}),tmp_path)
    def no_temporary(*args):
        raise AssertionError('zero transport must not allocate a temporary field')
    hydration.field_buffer=no_temporary
    hydration.populate()
    assert seen==[(('state','zero',tuple(shape),np.dtype('<f4' if kind=='real' else '<i4')),dict(nlevels=2))]
