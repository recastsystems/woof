from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.producer_cloud import (MYNN_CLOUD_FIELDS, cloud_authority,
    arwen_cloud_fields, normalize_cloud, publish_native_cloud, mean_cloud_contribution)
from woof.hex.diagnostic_export import diagnostic_export

@dataclass(frozen=True)
class Stream:
    name: str
    directions: tuple
    fields: tuple

def test_registry_publishing_is_conditional_complete_and_does_not_mutate():
    streams=[Stream('output',('output',),('theta',)),Stream('restart',('input','output'),('theta',))]
    rows=[{'name':n,'pool':'diag_physics'} for n,_,_ in MYNN_CLOUD_FIELDS]
    result=publish_native_cloud(streams,SimpleNamespace(fields=rows))
    assert result[0].fields == ('theta','cldfrac_bl','qc_bl','qi_bl')
    assert result[1] == streams[1]
    assert streams[0].fields == ('theta',)
    assert publish_native_cloud(streams,SimpleNamespace(fields=[])) is streams
    with pytest.raises(ValueError,match='incomplete'):
        publish_native_cloud(streams,SimpleNamespace(fields=rows[:1]))

@pytest.mark.parametrize('pbl',['ysu','bl_ysu'])
def test_ysu_structural_absence_is_not_a_zero_field(pbl):
    assert cloud_authority(pbl)=='structural_absence'
    assert arwen_cloud_fields(pbl)==()
    assert mean_cloud_contribution(pbl)==0.0

def test_mynn_mean_requires_actual_member_fields():
    assert mean_cloud_contribution('bl_mynn') is None

def test_unknown_pbl_is_not_silently_classified():
    with pytest.raises(ValueError,match='no native SGS'):
        cloud_authority('unknown')

class Seam:
    def __init__(self):
        words=np.array([0x80000000,0x33000001,0x38800002,0x3e000003],np.uint32)
        self.arrays={f'fields/{n}':words.view(np.float32).reshape(2,1,2).copy()
                     for n in arwen_cloud_fields('mynn')}
        self.arrays['fields/cldfra']=np.ones((2,1,2),np.float32)
        self._cp=SimpleNamespace(asnumpy=lambda x:np.array(x,copy=True))
    def _restart_manifest(self):return self.arrays.copy()
    def export_state(self):
        return {'identity':{},'scalars':{},'arrays':{k:self._cp.asnumpy(v) for k,v in self._restart_manifest().items()}}

def test_selective_public_export_preserves_all_sgs_words_and_excludes_total(monkeypatch):
    monkeypatch.setenv('WOOF_HEX_SELECTIVE_DIAGNOSTICS','1')
    seam=Seam();before={k:v.copy() for k,v in seam.arrays.items()}
    result,_=diagnostic_export(seam,arwen_cloud_fields('mynn'),selective=True)
    assert 'fields/cldfra' not in result['arrays']
    for public,native,_ in MYNN_CLOUD_FIELDS:
        source=seam.arrays[f'fields/{native}']
        actual=normalize_cloud(public,result['arrays'][f'fields/{native}'],2,2)
        np.testing.assert_array_equal(actual.view(np.uint32),source[:,0,:].view(np.uint32))
        np.testing.assert_array_equal(source.view(np.uint32),before[f'fields/{native}'].view(np.uint32))
    with pytest.raises(ValueError,match='shape'):
        normalize_cloud('qc_bl',np.ones((2,3),np.float32),2,2)


@pytest.mark.parametrize('include_cloud',[False, True])
@pytest.mark.parametrize('pbl',['mynnedmf','ysu'])
def test_whole_suite_snapshot_routes_native_3d_cloud_without_total_substitution(monkeypatch,pbl,include_cloud):
    import sys
    from woof.hex import cuda_suite_physics_v841 as module
    seam=Seam()
    for name in ('tsk','hfx','qfx','lh'):
        seam.arrays[f'fields/{name}']=np.ones((1,1,2),np.float32)
    for name in ('tslb','smois'):
        seam.arrays[f'fields/{name}']=np.ones((4,1,2),np.float32)
    seam.accumulated_precipitation=lambda:{'RAINNC':np.ones(2,np.float32),'RAINC':np.ones(2,np.float32)}
    seam.step_index=1;seam.elapsed_seconds=60.;seam.call_counts={}
    backend=object.__new__(module.PersistentSuiteCudaPhysicsBackendV841)
    backend._phase='boundary';backend._seam=seam
    backend._constructor=SimpleNamespace(selection={'pbl_scheme':pbl},n_levels=2,n_columns=2,n_soil_levels=4,physics=None)
    backend._profile={'microphysics':'wsm6'}
    backend._private_binding_guard=lambda:None
    backend._validate_surface_classification=lambda:{}
    backend._validate_noahmp_census=lambda **kwargs:None
    monkeypatch.setitem(sys.modules,'cupy',np)
    monkeypatch.setattr(module,'require_resident_array',lambda *args,**kwargs:None)
    monkeypatch.setattr(module,'history_cumulus_rain_rate',lambda *args:{})
    monkeypatch.setattr(module,'required_precipitation_keys',lambda **kwargs:{'RAINNC','RAINC'})
    snapshot=backend.diagnostic_snapshot(include_native_cloud=include_cloud)
    assert set(snapshot.cloud)==({'cldfrac_bl','qc_bl','qi_bl'} if include_cloud and pbl=='mynnedmf' else set())
    assert 'cldfra' not in snapshot.cloud
    for public,native,_ in MYNN_CLOUD_FIELDS:
        if public in snapshot.cloud:
            np.testing.assert_array_equal(snapshot.cloud[public].view(np.uint32),seam.arrays[f'fields/{native}'][:,0,:].view(np.uint32))
    assert not set(snapshot.cloud)&set(snapshot.surface)


@pytest.mark.parametrize("pbl, expected", [("mynnedmf", None), ("mynn", None), ("ysu", 0.0)])
def test_absent_mean_contribution_has_producer_authority(pbl, expected):
    from woof.hex.producer_cloud import mean_cloud_contribution
    assert mean_cloud_contribution(pbl) == expected
