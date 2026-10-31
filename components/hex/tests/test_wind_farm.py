"""Shared turbine input, native MYNN coupling and output contracts."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from woof.hex.wind_farm import WindFarmConfig, source_table, require_mynn, apply_native, attach_arwen, WindFarm


def test_off_has_no_input_or_backend_work():
    assert WindFarmConfig().wind_farms == 'off'
    assert source_table(None) is None
    assert source_table('off') is None
    apply_native(object(), {})
    attach_arwen(object(), {}, object())


@pytest.mark.parametrize('pbl', [0, 1, 'bl_ysu'])
def test_no_tke_coupling_is_named(pbl):
    with pytest.raises(ValueError, match='no coupled turbine TKE source'):
        require_mynn(pbl)


def test_held_pbl_is_refused():
    with pytest.raises(ValueError, match='held momentum tendency'):
        require_mynn('bl_mynn', interval='00:05:00')
    require_mynn('bl_mynn', interval='none')
    require_mynn(5, interval=0)


def test_native_hook_updates_the_live_mynn_carriers():
    fields={name:np.zeros((3,5),np.float32,order='F') for name in
            ('u','v','rho','qke','qke_adv','rublten','rvblten')}
    fields['delt']=18.
    interfaces=np.zeros((5,3),np.float32)
    class Farm:
        shape=(4,1,3)
        def apply_mynn(self, atmosphere, carriers, rates, *, step_seconds):
            assert step_seconds == 18.
            assert np.shares_memory(atmosphere['z_interface'], interfaces)
            carriers['qke'][:]+=2
            carriers['qke_adv'][:]+=2
            rates['du'][:]-=1
    provider=SimpleNamespace(wind_farm=Farm(),prepared=lambda name:interfaces)
    apply_native(provider,fields)
    for name in ('qke','qke_adv'):
        np.testing.assert_array_equal(fields[name][:,:4],2)
        np.testing.assert_array_equal(fields[name][:,4],0)
    np.testing.assert_array_equal(fields['rublten'][:,:4],-1)
    np.testing.assert_array_equal(fields['rublten'][:,4],0)


def test_history_indexes_are_independent_of_cell_count(monkeypatch):
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(asnumpy=lambda x:np.array(x,copy=True)))
    wind=WindFarm.__new__(WindFarm)
    fields={'POWER':np.zeros((1,3),np.float32),'WIND_TKE_SOURCE':np.zeros((4,1,3),np.float32),
            'TURBINE_POWER':np.asarray([1.,2.],np.float32),'FARM_POWER':np.asarray([3.],np.float32)}
    wind.farm=SimpleNamespace(shape=(4,1,3),history_fields=lambda:fields)
    result=wind.history_fields()
    assert result['POWER'][1:] == (('nCells',),'W')
    assert result['WIND_TKE_SOURCE'][0].shape == (4,3)
    assert result['TURBINE_POWER'][1] == ('wind_turbine',)
    assert result['FARM_POWER'][1] == ('wind_farm',)
    fields['TURBINE_POWER'][:]=10
    np.testing.assert_array_equal(result['TURBINE_POWER'][0],[1.,2.])


def test_whole_inventory_ids_do_not_exceed_a_netcdf_attribute(tmp_path):
    import hashlib
    import json
    wind = WindFarm.__new__(WindFarm)
    ids = [f'turbine-{i}' for i in range(70000)]
    wind.farm = SimpleNamespace(ids=ids, farm_ids=['farm-a', 'farm-b'],
        table_sha256='1' * 64, options={'tke_factor': .25},
        data={'provenance': {'license': 'CC0-1.0'}})
    wind.write_metadata(tmp_path)
    compact = wind.metadata().encode()
    assert len(compact) < 4096
    reference = json.loads(compact)
    payload = (tmp_path / reference['index_file']).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == reference['index_sha256']
    document = json.loads(payload)
    assert document['turbine_ids'] == ids
    assert document['farm_ids'] == ['farm-a', 'farm-b']
    assert document['input_provenance']['license'] == 'CC0-1.0'
    assert not (tmp_path / 'wind-farms.json.tmp').exists()
