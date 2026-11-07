"""Generic physics selections must honor the inherited resident-history interface."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from woof.hex import cuda_suite_physics_v841 as suite

@pytest.mark.parametrize('nsoil',[0,2,4])
def test_history_capture_matches_host_bytes_without_export_roundtrip(monkeypatch,nsoil):
    backend=object.__new__(suite.PersistentSuiteCudaPhysicsBackendV841)
    backend._phase='boundary'
    backend._constructor=SimpleNamespace(n_levels=2,n_columns=3,n_soil_levels=nsoil,physics={})
    backend._profile={'microphysics':'wsm6'}
    backend._private_binding_guard=lambda:None
    backend._validate_surface_classification=lambda:{}
    backend._validate_noahmp_census=lambda require:None
    values={'fields/q2':np.array([0x80000000,0xb6000000,0x7fc00042],dtype=np.uint32).view(np.float32)[None,None,:],
            'fields/tslb':np.arange(max(nsoil,4)*3,dtype=np.float32).reshape(max(nsoil,4),1,3)[:nsoil or 4],
            'diag/olr':np.full((1,3),240.,dtype=np.float32)}
    copies=[]
    class Seam:
        pending=False
        step_index=0
        elapsed_seconds=0.
        call_counts={}
        _cp=SimpleNamespace(asnumpy=lambda v:(copies.append(v),np.array(v,copy=True))[1])
        def _restart_manifest(self):return dict(values)
        def export_state(self):return {'identity':{},'arrays':{k:self._cp.asnumpy(v) for k,v in values.items()},'scalars':{}}
        def accumulated_precipitation(self):return {'RAINC':np.zeros(3,dtype=np.float32)}
    backend._seam=Seam()
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(asarray=np.asarray,ascontiguousarray=np.ascontiguousarray,array=np.array,zeros=np.zeros,float32=np.float32))
    monkeypatch.setattr(suite,'_REQUIRED_ARWEN_EXPORT_FIELDS',('q2',))
    monkeypatch.setattr(suite,'_OPTIONAL_ARWEN_EXPORT_FIELDS',('tslb','olr'))
    monkeypatch.setattr(suite,'require_resident_array',lambda *a,**k:None)
    monkeypatch.setattr(suite,'required_precipitation_keys',lambda **k:{'RAINC'})
    monkeypatch.delenv('WOOF_HEX_SELECTIVE_DIAGNOSTICS',raising=False)
    before={k:v.tobytes() for k,v in values.items()}
    host=backend.diagnostic_snapshot()
    copies.clear()
    history=backend.history_diagnostic_snapshot()
    assert not copies
    for group in ['surface','soil','precipitation','gwdo']:
        a,b=getattr(host,group),getattr(history,group)
        assert set(a)==set(b)
        for key in a:
            assert a[key].shape==b[key].shape
            assert a[key].dtype==b[key].dtype
            assert a[key].tobytes()==b[key].tobytes()
    assert history.receipt['selected_export_h2d_bytes']==0
    assert history.receipt['selected_export_hashes']=={}
    assert before=={k:v.tobytes() for k,v in values.items()}
    backend._phase='candidate'
    with pytest.raises(RuntimeError,match='committed boundary'):backend.history_diagnostic_snapshot()


def test_parent_history_call_binds_on_the_suite_seam():
    """The parent history path calls diagnostic_snapshot(selective=True, resident=True); the hex 0.4.4 suite
    seam took no keywords and every column-suite run died with a TypeError at its hour-0 frame. The keywords
    bind, and off a committed boundary the body refuses with its own RuntimeError, which proves the call bound."""
    import inspect
    inspect.signature(suite.PersistentSuiteCudaPhysicsBackendV841.diagnostic_snapshot).bind(
        object(), selective=True, resident=True)
    backend = object.__new__(suite.PersistentSuiteCudaPhysicsBackendV841)
    backend._phase = "phase1"
    with pytest.raises(RuntimeError, match="committed boundary"):
        backend.diagnostic_snapshot(selective=True, resident=True)
