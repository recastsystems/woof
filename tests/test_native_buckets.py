"""Integer bucket carriers survive real checkpoint serialization without rounding."""
import numpy as np
from woof.core import output_diagnostics as output
from woof.core.misc_buckets import COUNTERS
from woof.io import restart
from test_restart import _cfg,_shim_state,_fill_setup

def test_integer_counter_checkpoint_roundtrip(monkeypatch,tmp_path):
    cfg=_cfg(bucket_mm=.001,bucket_j=100000.)
    src=_shim_state(cfg,monkeypatch);_fill_setup(src);patterns={}
    for i,name in enumerate((*COUNTERS,'bucket_epoch')):
        a=src.existing_scratch(name)
        if name!='bucket_epoch':
            assert a.dtype==np.int32
            a[...] = np.arange(a.size,dtype='i4').reshape(a.shape)+16777217+i
        else:a[...] = (3600.25,1.e-6)
        patterns[name]=a.tobytes()
    path=restart.write_restart(tmp_path/'bucket.npz',src,cfg)
    dst=_shim_state(cfg,monkeypatch);_fill_setup(dst);restart.restore_restart(path,dst,cfg)
    for name,data in patterns.items():assert dst.existing_scratch(name).tobytes()==data
    with np.load(path,allow_pickle=False) as archive:
        for name in COUNTERS:assert archive['scratch/'+name].dtype==np.int32

def test_bucket_native_sources_schema_and_inert_identity():
    from woof.ingest import wrfrst
    from woof.io.wrf_output_schema import OUTPUT_FIELDS_BY_NETCDF_NAME
    from woof.core.misc_buckets import drop_inert
    for name in COUNTERS:
        assert wrfrst.WRFRST_DISPOSITIONS[name.upper()].kind=='read'
        assert wrfrst.scratch_restore_plan(name,0)==('fill',name.upper())
        assert OUTPUT_FIELDS_BY_NETCDF_NAME[name.upper()].dtype=='i4'
    assert not set(COUNTERS)&set(output.scratch_shapes(_cfg()))
    d={'bucket_mm':-1.,'bucket_j':-1.,'mp_physics':8};drop_inert(d)
    assert d=={'mp_physics':8}
