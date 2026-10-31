"""ABI copy/selection checks; no CUDA science or forecast identity claimed."""
from types import SimpleNamespace, ModuleType
from contextlib import nullcontext
import sys
import numpy as np
import pytest

from woof.hex.mpasphys import SlotRegistry, resolve_namelist, SlotCall, StepFlags, Backend
from woof.hex.mpasphys.device_pools_v842 import NativeDevicePoolViews, register_cuda_slots


@pytest.fixture(autouse=True)
def host_only_memory_release(monkeypatch):
    # These ABI checks use NumPy stand-ins. Pool-release hooks otherwise import
    # CuPy even though no test allocates a device array or claims CUDA coverage.
    monkeypatch.setattr('woof.hex.mpasphys.device_pools_v842.phase_release',
                        lambda *args, **kwargs: nullcontext())


class Native:
    def __init__(self):
        self.fields={('diag','aliased',1):np.array([1,-0.,9,777],np.float32),
                     ('state','q',1):np.array([[1,2,3,888]],np.float32),
                     ('state','q',2):np.array([[4,5,6,888]],np.float32)}
        self._fields={('diag','aliased'):(np.dtype('float32'),(4,),1),
                      ('state','q'):(np.dtype('float32'),(1,4),2)}
        self.variables={'u_p':(np.arange(6,dtype=np.float32).reshape(3,2,1),(1,1,1)),
                        'xice_threshold':(np.asarray(.2,np.float32),())}
        self.abi={'variables':{'u_p':{'rank':3},'xice_threshold':{'rank':0}},
                  'constants':{'gravity':{}}}
    def get_field(self,pool,name,level=1):return self.fields[pool,name,level].copy()
    def put_field(self,pool,name,value,level=1,nlevels=1):self.fields[pool,name,level]=value.copy()
    def get_variable(self,name):
        a,lower=self.variables[name];return a.copy(),lower
    def put_variable(self,name,value,lower):self.variables[name]=value.copy(),lower
    def get_constant(self,name):
        if name!='gravity':raise KeyError(name)
        return np.float32(9.81)


def provider():
    fields=[dict(pool='diag',name='registry_alias',pool_name='aliased',dimensions=['nCells']),
            dict(pool='state',name='q',pool_name='q',dimensions=['nVertLevels','nCells'])]
    layout=SimpleNamespace(pools={'diag','state','mesh','atm_input'},fields=fields,
                           dimensions={'nCellsSolve':3},configs={'config_do_restart':False})
    class DeviceArray(np.ndarray):
        pass
    def contiguous(value):
        # CuPy 14.2 accepts device arrays here, not a NumPy host allocation.
        assert isinstance(value, DeviceArray)
        return np.array(value,copy=True,order='C').view(DeviceArray)
    cp=SimpleNamespace(asarray=lambda a:np.asarray(a).view(DeviceArray),
                       ascontiguousarray=contiguous)
    return NativeDevicePoolViews(SimpleNamespace(native=Native(),layout=layout),cp)


def call(slot='surface_layer',scheme='sf_mynn'):
    return SlotCall(slot,scheme,Backend.HEX_CUDA,1,0.,30.,StepFlags(),None,{})


def test_aliases_share_device_storage_and_only_solve_words_return():
    p=provider();p.begin(call(),{})
    a=p['diag']['registry_alias'];assert a is p['diag']['aliased']
    assert a.view(np.uint32)[1]==0x80000000
    a[0]=np.float32(42)
    p.commit()
    assert p.native.fields['diag','aliased',1].tolist()==[42,-0.,9,777]


def test_persistent_fields_do_not_pin_idle_scratch_or_drain_on_cached_lookup(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_MEMBER_MEMORY", "shared-card")
    p = provider()
    drains = []
    p.cp.get_default_memory_pool = lambda: SimpleNamespace(
        free_all_blocks=lambda: drains.append(True))
    p.resident = True
    field = p.field('diag', 'aliased')
    prepared = p.prepared('u_p')
    assert len(drains) == 2
    p.buffers.clear()
    assert p.field('diag', 'registry_alias') is field
    assert p.prepared('u_p') is prepared
    assert len(drains) == 2


def test_microphysics_level_and_prepared_fortran_views_preserve_halos():
    p=provider();p.begin(call('microphysics','mp_wsm6'),{'profiles':{'u3d':'u_p'}})
    q=p['state']['q'];assert q.tolist()==[[4,5,6]]
    q[0,0]=np.float32(8)
    u=p['atm_input']['u3d'];assert u.shape==(2,3) and u.flags.c_contiguous
    u[1,2]=np.float32(29)
    p.commit()
    assert p.native.fields['state','q',1].tolist()==[[1,2,3,888]]
    assert p.native.fields['state','q',2].tolist()==[[8,5,6,888]]
    assert p.native.variables['u_p'][0][2,1,0]==np.float32(29)


def test_native_scalar_and_constant_aliases_keep_rkind():
    p=provider();p.begin(call(),{'constant_aliases':{'g':'gravity'}})
    assert type(p['scalar_state']['xice_threshold']) is np.float32
    assert p['constants']['g'].tobytes()==np.float32(9.81).tobytes()
    p.native.get_constant=None
    with pytest.raises(RuntimeError,match='constant export is absent'):p['constants']['g']


def test_table_selected_callback_commits_through_real_registry(monkeypatch):
    p=provider();module=ModuleType('woof.hex.mpasphys.oml_v842');seen=[]
    def callback(c):
        seen.append(c.scheme);c.pools['diag']['aliased'][0]=np.float32(11)
    module.launch_slot=callback;monkeypatch.setitem(sys.modules,module.__name__,module)
    config=resolve_namelist('&physics\nconfig_physics_suite="mesoscale_reference"\nconfig_oml1d=.true.\n/',backends={'oml':'hex-cuda'})
    r=SlotRegistry();register_cuda_slots(r,config,p)
    r.run(call('oml','oml1d'))
    assert seen==['oml1d'] and p.native.fields['diag','aliased',1][0]==11
    assert len(r._callbacks)==1


def test_missing_selected_module_or_descriptor_is_explicit(monkeypatch):
    p=provider();r=SlotRegistry()
    config=resolve_namelist('&physics\nconfig_physics_suite="none"\nconfig_radt_lw_scheme="cam_lw"\n/',backends={'radiation_lw':'hex-cuda'})
    with pytest.raises(RuntimeError,match='published callback descriptor'):register_cuda_slots(r,config,p)


@pytest.mark.parametrize('scheme', ['sf_monin_obukhov', 'sf_monin_obukhov_rev'])
def test_selected_mm5_diagnostics_read_committed_post_surface_words(monkeypatch,scheme):
    p=provider();p.session.hooks=SimpleNamespace(external={})
    module=ModuleType('woof.hex.mpasphys.sfclay_mm5_v842');events=[]
    module.bind_native_pools=lambda owner,c:owner
    def surface(c):
        c.pools['diag']['aliased'][0]=np.float32(11)
        events.append('surface')
    def diagnostics(owner,c):
        owner.begin(c,{})
        a=owner['diag']['aliased']
        assert a[0]==np.float32(11)
        a[0]=np.float32(21);owner.commit();events.append('diagnostics')
    module.run_slot=surface;module.run_native_diagnostics=diagnostics
    monkeypatch.setitem(sys.modules,module.__name__,module)
    config=resolve_namelist('&physics\nconfig_physics_suite="mesoscale_reference"\n'
        f'config_sfclayer_scheme="{scheme}"\n/',backends={'surface_layer':'hex-cuda'})
    registry=SlotRegistry();register_cuda_slots(registry,config,p)
    registry.run(call('surface_layer',scheme))
    p.session.hooks.external['atmphys_sfc_diagnostics'](call('atmphys_sfc_diagnostics','driver'))
    assert events==['surface','diagnostics']
    assert p.native.fields['diag','aliased',1].tolist()==[21,-0.,9,777]
    assert len(registry._callbacks)==1


def test_named_surface_shared_arrays_retain_native_halo_and_aliases():
    p=provider()
    p.native.variables['chs_sea']=(np.array([[1],[2],[-0.],[777]],np.float32),(1,1))
    p.begin(call('seaice','sf_noah'),{'profile_pools':{'seaice_shared':{'chs_sea':'chs_sea','t3d':'u_p'}}})
    shared=p['seaice_shared'];a=shared['chs_sea']
    assert a.shape==(3,) and a.view(np.uint32)[2]==0x80000000
    assert a is p['prepared']['chs_sea']
    assert shared['t3d'].shape==(2,3)
    a[0]=np.float32(42);p.commit()
    assert p.native.variables['chs_sea'][0][:,0].tolist()==[42,2,-0.,777]


@pytest.mark.parametrize('selector,module_name,slot,loader_name,loader_call,attribute',[
    ('mp_thompson','thompson_v842','microphysics','table_resources_v842','load_thompson_resources','thompson_v842_resources'),
    # The aerosol-aware entry binds its own loader: the classic Thompson auxiliary carries the unit
    # CCN table the aerosol loader refuses (fix install-physics-2).
    ('mp_thompson_aerosols','thompson_aero_v842','microphysics','thompson_aero_v842','load_resources','thompson_v842_resources'),
    ('bl_mynn','bl_mynn_v842','pbl','bl_mynn_v842','load_mynn_resources','mynn_v842_resources'),
    ('rrtmg_sw','rrtmg_sw_v842','radiation_sw','rrtmg_sw_v842','load_shortwave_resources','rrtmg_sw_v842_resources'),
])
def test_descriptor_installs_persistent_resources_before_binding(monkeypatch,tmp_path,selector,module_name,slot,loader_name,loader_call,attribute):
    import json
    p=provider();module=ModuleType('woof.hex.mpasphys.'+module_name);events=[]
    resources=module if loader_name==module_name else ModuleType('woof.hex.mpasphys.'+loader_name)
    def load(packet,device):
        events.append(('resources',packet));return {'tables':{'token':42},'device_math_source':'shared'}
    setattr(resources,loader_call,load)
    def bind(provider,c):
        assert getattr(provider,attribute)['tables']['token']==42
        events.append(('bind',c.slot));return {'q':provider['state']['q']}
    def run(c):
        c.pools['q'][0,0]=np.float32(17);events.append(('run',c.slot))
    module.bind_native_pools=bind;module.run_slot=run;module.shortwave_slot=run
    monkeypatch.setitem(sys.modules,module.__name__,module)
    monkeypatch.setitem(sys.modules,resources.__name__,resources)
    options={'microphysics':'config_microp_scheme','pbl':'config_pbl_scheme',
             'radiation_sw':'config_radt_sw_scheme'}
    config=resolve_namelist('&physics\nconfig_physics_suite="convection_permitting"\n'
        f'{options[slot]}="{selector}"\n/',backends={slot:'hex-cuda'})
    path=tmp_path/'resources.json';path.write_text(json.dumps({selector:{'packet':'certified'}}))
    # No packet given: first use builds or reuses the install's packet
    # (resources_build.ensure_packet) instead of refusing.
    built=[]
    def ensure(*,landuse=None,cache=None,include_mynnedmf=True):
        assert not include_mynnedmf
        built.append(landuse);return path
    monkeypatch.setattr('woof.hex.mpasphys.resources_build.ensure_packet',ensure)
    register_cuda_slots(SlotRegistry(),config,p)
    assert built==[None] and events==[('resources',{'packet':'certified'})]
    events.clear()
    missing=tmp_path/'other.json';missing.write_text(json.dumps({'other_scheme':{}}))
    with pytest.raises(RuntimeError,match=f'has no {selector} entry'):register_cuda_slots(SlotRegistry(),config,p,missing)
    registry=SlotRegistry();register_cuda_slots(registry,config,p,path)
    registry.run(call(slot,selector));registry.run(call(slot,selector))
    assert events==[('resources',{'packet':'certified'}),('bind',slot),('run',slot),
                    ('bind',slot),('run',slot)]
    level=2 if slot=='microphysics' else 1
    original=[4,5,6,888] if level==2 else [1,2,3,888]
    assert p.native.fields['state','q',level].tolist()==[[17,*original[1:]]]
    other=1 if level==2 else 2
    unchanged=[1,2,3,888] if other==1 else [4,5,6,888]
    assert p.native.fields['state','q',other].tolist()==[unchanged]


def test_rrfs_rank_four_prepared_views_round_trip_without_touching_padding():
    p = provider()
    original = np.arange(24, dtype=np.float32).reshape(4, 3, 1, 2)
    p.native.variables['spectral_p'] = (original.copy(), (1, 1, 1, 1))
    p.begin(call('radiation_sw', 'rrtmg_sw'), {})
    packed = p['prepared']['spectral_p']
    assert packed.shape == (2, 3, 3)
    np.testing.assert_array_equal(packed, original[:3, :, 0, :].transpose(2, 1, 0))
    packed[1, 2, 1] = np.float32(-7)
    p.commit()
    expected = original.copy()
    expected[1, 2, 0, 1] = np.float32(-7)
    np.testing.assert_array_equal(p.native.variables['spectral_p'][0], expected)


def test_rrfs_resident_halo_transaction_commits_before_exchange_and_refreshes():
    p = provider()
    events = []
    class Transfer(np.ndarray):
        def get(self):
            return np.array(self, copy=True)
    p.cp.asarray = lambda x: np.asarray(x).view(Transfer)
    p.cp.ascontiguousarray = lambda x: np.array(x, copy=True, order='C').view(Transfer)
    p.cp.cuda = SimpleNamespace(get_current_stream=lambda: SimpleNamespace(
        synchronize=lambda: events.append('sync')))
    p.native._registered = {'registry_alias': ('diag', 'aliased')}
    p.native._groups = {}
    def register(name, fields, level, halo_layers):
        p.native._groups[name] = (tuple(fields), level, tuple(halo_layers))
    def exchange(name):
        assert p.native.fields['diag', 'aliased', 1][0] == np.float32(42)
        events.append('exchange')
        p.native.fields['diag', 'aliased', 1][-1] = np.float32(999)
    p.native.exchange_group = register
    p.native.exchange_halo_group = exchange
    descriptor = {'halo_groups': [{'name': 'physics:test', 'fields': ['registry_alias'],
                                   'time_level': 1, 'halo_layers': [2]}]}
    p.begin(call('convection', 'cu_grell_freitas_li'), descriptor)
    packed = p.resident_pool('diag')['registry_alias']
    assert packed.shape == (4,)
    packed[0] = np.float32(42)
    p.exchange_halo_group('physics:test')
    assert events == ['sync', 'exchange']
    assert packed[-1] == np.float32(999)
    assert p.native._groups['physics:test'] == (('registry_alias',), 1, (2,))
    p.commit()
    assert p.native.fields['diag', 'aliased', 1].tolist() == [42, -0., 9, 999]


def extent_provider(*, persistent, entity='nCells', owned=3, padded=True):
    """Native storage with owned cells, real halos and a separate final dummy."""
    p = provider()
    p.resident = persistent
    count = 6 if padded else owned
    words = np.resize(np.array([0x80000000, 0x7fc12345, 0x3f800000,
                                0x40800000, 0x40a00000, 0x7fc54321], np.uint32), (2, count))
    p.native.fields['state', 'q', 1] = words.view(np.float32).copy()
    p.native.fields['state', 'q', 2] = words.view(np.float32).copy()
    p.native._fields['state', 'q'] = (np.dtype('float32'), (2, count), 2)
    p.session.layout.fields[1].update(name='registry_q', dimensions=['nVertLevels', entity])
    p.session.layout.dimensions.update({entity: count - int(padded), entity + 'Solve': owned})
    class Device(np.ndarray):
        def get(self, order='C'):
            return np.array(self, order=order, copy=True)
    p.cp.asarray = lambda value: np.array(value, copy=True).view(Device)
    p.cp.ascontiguousarray = lambda value: np.ascontiguousarray(value).copy().view(Device)
    p.cp.ndarray = Device
    p.cp.float32, p.cp.int32, p.cp.uint32 = np.float32, np.int32, np.uint32
    p.test_reads = []
    original_get = p.native.get_field
    def get_field(pool, name, level=1):
        p.test_reads.append((pool, name, level))
        return original_get(pool, name, level)
    p.native.get_field = get_field
    return p


@pytest.mark.parametrize('entity', ['nCells', 'nEdges', 'nVertices'])
@pytest.mark.parametrize('persistent', [False, True])
@pytest.mark.parametrize('full_first', [False, True])
@pytest.mark.parametrize('first_name,second_name', [('registry_q', 'q'), ('q', 'registry_q')])
def test_mixed_extent_requests_refuse_without_losing_authority(
        entity, persistent, full_first, first_name, second_name):
    p = extent_provider(persistent=persistent, entity=entity)
    original = p.native.fields['state', 'q', 1].copy()
    p.begin(call(), {})
    packed = p.field('state', first_name, resident=full_first)
    expected_shape = (2, 6 if full_first else 3)
    assert packed.shape == expected_shape and packed.flags.c_contiguous
    packed.view(np.uint32)[..., 0] = 0x40000000
    if full_first:
        # Explicit full storage includes real halos and the native dummy.
        packed.view(np.uint32)[..., -2] = 0x40400000
        packed.view(np.uint32)[..., -1] = 0x80000000
    key = ('state', 'q', 1)
    authority = p.buffers[key]
    reads = list(p.test_reads)
    with pytest.raises(ValueError, match=r'cached extent conflict for state.q level 1'):
        p.field('state', second_name, resident=not full_first)
    assert p.buffers[key] is authority
    assert p.field('state', second_name, resident=full_first) is packed
    assert p.test_reads == reads == [key]
    assert np.array_equal(p.native.fields[key].view(np.uint32), original.view(np.uint32))
    expected = original.copy()
    expected[..., :packed.shape[-1]] = packed
    p.commit()
    if persistent:
        assert p.resident_buffers[key] is authority
        assert np.array_equal(p.native.fields[key].view(np.uint32), original.view(np.uint32))
        p.publish()
        assert p.transfers == {'initial_upload_bytes': packed.nbytes,
                               'publication_download_bytes': packed.nbytes}
    assert np.array_equal(p.native.fields[key].view(np.uint32), expected.view(np.uint32))
    if not full_first:
        # Neither halo nor dummy words can change through owned publication.
        assert np.array_equal(p.native.fields[key][..., 3:].view(np.uint32),
                              original[..., 3:].view(np.uint32))


@pytest.mark.parametrize('full_first', [False, True])
def test_persistent_extent_refusal_survives_retirement_begin_and_commit(full_first):
    p = extent_provider(persistent=True)
    p.begin(call(), {})
    packed = p.field('state', 'registry_q', resident=full_first)
    packed[0, 0] = 42
    key = ('state', 'q', 1)
    authority = p.resident_buffers[key]
    p.buffers.pop(key)  # Input retirement must not erase persistent extent ownership.
    p.commit()
    p.begin(call('pbl', 'bl_mynn'), {})
    assert key not in p.buffers
    with pytest.raises(ValueError, match='cached extent conflict'):
        p.field('state', 'q', resident=not full_first)
    assert key not in p.buffers
    assert p.resident_buffers[key] is authority
    assert p.field('state', 'q', resident=full_first) is packed
    p.publish()
    p.begin(call(), {})
    with pytest.raises(ValueError, match='cached extent conflict'):
        p.field('state', 'registry_q', resident=not full_first)
    assert p.test_reads == [key]
    assert p.native.fields[key][0, 0] == 42


def test_owned_write_field_preserves_halo_dummy_levels_and_extent_after_publish():
    p = extent_provider(persistent=True)
    p.begin(call(), {})
    initial_one = p.native.fields['state', 'q', 1].copy()
    initial_two = p.native.fields['state', 'q', 2].copy()
    one = p.field('state', 'registry_q')
    p.write_field('state', 'q', np.full((2, 4), 8, np.float32), level=2)
    assert p.level == 1
    assert np.array_equal(one.view(np.uint32), initial_one[..., :3].view(np.uint32))
    p.begin(call('microphysics', 'mp_thompson'), {})
    two = p.field('state', 'registry_q')
    assert np.all(two == 8) and two.flags.c_contiguous
    with pytest.raises(ValueError, match='cached extent conflict'):
        p.resident_pool('state')['q']
    p.publish()
    assert np.array_equal(p.native.fields['state', 'q', 1].view(np.uint32), initial_one.view(np.uint32))
    assert np.array_equal(p.native.fields['state', 'q', 2][..., 3:].view(np.uint32),
                          initial_two[..., 3:].view(np.uint32))
    assert np.all(p.native.fields['state', 'q', 2][..., :3] == 8)


def test_full_cache_rejects_owned_write_without_mutation_and_restores_level():
    p = extent_provider(persistent=True)
    p.begin(call(), {})
    p.level = 2
    full = p.resident_pool('state')['registry_q']
    before = full.copy()
    p.level = 1
    with pytest.raises(ValueError, match='cached extent conflict'):
        p.write_field('state', 'q', np.zeros((2, 3), np.float32), level=2)
    assert p.level == 1
    assert np.array_equal(full.view(np.uint32), before.view(np.uint32))


@pytest.mark.parametrize('full_first', [False, True])
def test_transaction_commit_allows_extent_change_with_fresh_native_copy(full_first):
    p = extent_provider(persistent=False)
    p.begin(call(), {})
    first = p.field('state', 'registry_q', resident=full_first)
    first[0, 0] = 17
    p.commit()
    p.begin(call(), {})
    second = p.field('state', 'q', resident=not full_first)
    assert second.shape == (2, 3 if full_first else 6)
    assert second[0, 0] == 17 and second is not first
    assert p.test_reads == [('state', 'q', 1), ('state', 'q', 1)]


@pytest.mark.parametrize('count', [3, 5])
def test_direct_buffer_writers_keep_tuple_contract_and_extent_validation(count):
    p = extent_provider(persistent=False)
    p.begin(call(), {})
    # Noah seeds owned words; ozone initialization seeds all real words without dummy.
    host = p.native.get_field('state', 'q')
    packed = p.cp.ascontiguousarray(p.cp.asarray(host[..., :count]))
    entry = (host, packed, (..., slice(0, count)))
    p.buffers['state', 'q', 1] = entry
    if count == 3:
        assert p.field('state', 'registry_q') is packed
    else:
        with pytest.raises(ValueError, match='cached extent conflict'):
            p.field('state', 'registry_q')
    with pytest.raises(ValueError, match='cached extent conflict'):
        p.field('state', 'q', resident=True)
    assert p.buffers['state', 'q', 1] is entry
    assert p.test_reads == [('state', 'q', 1)]


def test_equivalent_extents_and_nonentity_fields_share_storage():
    p = extent_provider(persistent=True, padded=False)
    p.begin(call(), {})
    owned = p.field('state', 'registry_q')
    assert p.field('state', 'q', resident=True) is owned
    p.session.layout.fields.append(dict(pool='diag', name='constant_alias',
        pool_name='constant', dimensions=[]))
    p.native.fields['diag', 'constant', 1] = np.asarray(-0., np.float32)
    p.native._fields['diag', 'constant'] = (np.dtype('float32'), (), 1)
    full = p.field('diag', 'constant_alias', resident=True)
    assert full.shape == ()
    assert full.view(np.uint32).item() == 0x80000000
    assert p.field('diag', 'constant') is full
    p.commit()
    p.begin(call(), {})
    assert p.field('state', 'registry_q', resident=True) is owned
    assert p.field('diag', 'constant_alias') is full
