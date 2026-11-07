"""Retained CAM absorption and its clock survive the generic checkpoint hooks."""
from pathlib import Path
from types import SimpleNamespace
import json,numpy as np,pytest
from woof.core.cam3 import Cam3Radiation
from woof.io import restart

def scheme():
    item=object.__new__(Cam3Radiation)
    item._metadata=json.loads(Path('woof/core/kernels/cam3_emissivity.json').read_text())['camrad']['parameters']
    item._retained={};item._abs_time=None;item.update_count=0;item._state=None
    item.latitude_deg=np.zeros((1,2),np.float32)
    item._ozone=np.zeros((2,59*13),np.float32)
    return item

def live():
    item=scheme();item._abs_time=0.;item.update_count=2
    for field in item._metadata:
        if field['name'] in item._retained_names:
            size=eval(field['count_expression'],{'CAM_LEVELS':48})
            item._retained[field['name']]=np.arange(2*size,dtype=np.float32).reshape(1,2,size).transpose(2,0,1).copy()
    return item

def test_checkpoint_manifest_restores_absorption_arrays_and_clock():
    source=live();driver=SimpleNamespace(radiation_callable=source)
    manifest=restart._radiation_owned_manifest(driver)
    target=scheme();fresh=SimpleNamespace(radiation_callable=target)
    [(adapter,payload)]=restart._radiation_owned_payloads(manifest,fresh)
    state=SimpleNamespace(p=np.empty((48,1,2),np.float32))
    adapter.checkpoint_restore(payload,state,np.asarray)
    assert target._abs_time==0. and target.update_count==2
    for name,value in source._retained.items():
        assert target._retained[name].tobytes()==value.tobytes()
        assert target._retained[name].flags.c_contiguous
    restart._callable_state_check(target,restart.RADIATION_CALLABLE_ARRAYS,restart.RADIATION_CALLABLE_CONTAINERS,'radiation')

def test_checkpoint_rejects_missing_held_absorption_and_extra_arrays():
    source=live();payload=source.checkpoint_manifest();payload.pop('absnxt_3d')
    with pytest.raises(ValueError,match='retained absorption'):source.checkpoint_validate(payload,SimpleNamespace(p=np.empty((48,1,2),np.float32)))
    source._retained['new_field']=np.zeros(1,np.float32)
    with pytest.raises(ValueError,match='all three'):source.checkpoint_manifest()

def test_checkpoint_rejects_state_for_an_absent_component():
    with pytest.raises(restart.RestartMismatchError,match='absent radiation'):
        restart._radiation_owned_payloads({'radiation/owned/1/absorption_clock':np.zeros(2)},SimpleNamespace(radiation_callable=scheme()))


def test_member_namespace_accepts_only_validated_radiation_payload():
    source=live();driver=SimpleNamespace(radiation_callable=source)
    state=SimpleNamespace(p=np.empty((48,1,2),np.float32))
    payload=restart._radiation_owned_manifest(driver)
    restart._validate_member_namespaces(payload,state,driver,'owned.npz',restart.RESTART_FORMAT_VERSION)
    payload['radiation/owned/0/unknown']=np.zeros(1,np.float32)
    with pytest.raises(restart.RestartMismatchError,match='retained absorption'):
        restart._validate_member_namespaces(payload,state,driver,'owned.npz',restart.RESTART_FORMAT_VERSION)


def test_streaming_keeps_horizontal_cache_planes_and_restores_clock():
    cp=pytest.importorskip('cupy')
    source=live();driver=SimpleNamespace(radiation_callable=source)
    manifest=restart._radiation_owned_manifest(driver,streaming=True)
    assert len(manifest)==4
    for name,array in manifest.items():
        assert array.shape[-2:]==(1,2) and array.flags.c_contiguous
        if not name.endswith('/absorption_clock'):
            assert array is source._retained[name.rsplit('/',1)[1]]
    target=scheme();target.streaming_restore_scalars(source.streaming_scalars())
    assert target._abs_time==0. and target.update_count==2
    clock=manifest['radiation/owned/0/absorption_clock']
    source.streaming_restore_scalars({'abs_time':1800.,'update_count':3})
    assert source.streaming_manifest()['absorption_clock'] is clock
    np.testing.assert_array_equal(cp.asnumpy(clock[:,0,0]),[1800.,3.])
    cold=scheme();payload=cold.streaming_manifest(state_shape=(16,1,2))
    assert set(cold._retained)==set(cold._retained_names)
    assert all(a.shape[-2:]==(1,2) and a.flags.c_contiguous for a in cold._retained.values())
    payload={name:cp.asnumpy(value) for name,value in payload.items()}
    state=SimpleNamespace(p=np.empty((16,1,2),np.float32))
    fresh=scheme();fresh.checkpoint_restore(payload,state,np.asarray)
    assert fresh._abs_time is None and fresh.update_count==0
    assert set(fresh._retained)==set(fresh._retained_names)
