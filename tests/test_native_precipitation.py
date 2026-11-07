"""Native precipitation amounts and restart-relative clocks share one lifecycle."""
import numpy as np
from woof.core import output_diagnostics as output
from woof.io import restart
from test_restart import _cfg,_shim_state,_fill_setup

def test_nonzero_restart_origin_and_amounts_roundtrip(monkeypatch,tmp_path):
    cfg=_cfg(prec_acc_dt=60.)
    src=_shim_state(cfg,monkeypatch);_fill_setup(src)
    patterns={}
    for i,name in enumerate(('prec_acc_c','prec_acc_nc','snow_acc_nc','graup_acc_nc','prec_acc_epoch')):
        a=src.existing_scratch(name)
        a[...] = np.arange(a.size,dtype='f4').reshape(a.shape)+np.float32(i+.25)
        patterns[name]=a.tobytes()
    path=restart.write_restart(tmp_path/'precip.npz',src,cfg)
    dst=_shim_state(cfg,monkeypatch);_fill_setup(dst)
    restart.restore_restart(path,dst,cfg)
    for name,data in patterns.items():assert dst.existing_scratch(name).tobytes()==data
    with np.load(path,allow_pickle=False) as archive:
        assert all('scratch/'+name in archive.files for name in patterns)

def test_native_restart_sources_and_inert_allocation():
    from woof.ingest import wrfrst
    for name in ('prec_acc_c','prec_acc_nc','snow_acc_nc'):
        assert wrfrst.WRFRST_DISPOSITIONS[name.upper()].kind=='read'
        assert wrfrst.scratch_restore_plan(name,0)==('fill',name.upper())
    assert 'prec_acc_epoch' not in output.scratch_shapes(_cfg())
    assert 'prec_acc_epoch' not in output.scratch_shapes(_cfg(prec_acc_dt1=60.))
