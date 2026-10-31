"""Word-level framework replay, independent of CUDA or a Fortran runtime."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import StepFlags, resolve_namelist
from woof.hex.mpasphys.framework_backend_v842 import FrameworkPhysics, registry_source
from woof.hex.mpasphys.framework_calendar_v842 import ModelTime, PhysicsClock
from woof.hex.mpasphys.framework_init_v842 import FrameworkInitialization
from woof.hex.mpasphys.pool_v842 import NativePoolHydration, RegistryLayout


def test_registry_is_complete_without_source_tree():
    text = '&physics\nconfig_physics_suite="convection_permitting"\n/'
    layout = RegistryLayout(registry_source(), resolve_namelist(text), text,
        dict(nCells=2,nEdges=3,nVertices=2,maxEdges=6,maxEdges2=12,
             vertexDegree=3,nVertLevels=4))
    state = next(r for r in layout.fields if r['name'] == 'scalars')
    assert state['shape'] == [8, 4, 3]
    assert state['components'] == ['qv','qc','qr','qi','qs','qg','ni','nr']
    assert layout.indices['state','index_nc'] == -1
    assert len(layout.fields) > 500


def test_restored_private_words_are_adopted_only_before_a_cold_view_exists():
    bits=np.array([[0x7fa12345,0x80000000,1]],np.uint32)
    words=bits.view(np.float32).copy(order='F')
    loaded=[]
    def cold():
        loaded.append(True)
        return np.zeros_like(words,order='F')
    with FrameworkPhysics() as native:
        native.put_deferred_field('state','owned',words.shape,words.dtype,cold,nlevels=2)
        native.adopt_restored_field('state','owned',words,level=1,nlevels=2)
        assert not loaded
        assert native.field_storage('state','owned',level=1) is words
        # A live second-level view must continue observing the ordinary copy.
        view=native.field_storage('state','owned',level=2)
        assert loaded == [True]
        native.adopt_restored_field('state','owned',words.copy(order='F'),level=2,nlevels=2)
        assert native.field_storage('state','owned',level=2) is view
        assert view.tobytes()==words.tobytes()
        assert not np.shares_memory(view,words)
        with pytest.raises(ValueError,match='shape'):
            native.adopt_restored_field('state','owned',words[:,:2],level=2,nlevels=2)


def test_deferred_cold_fields_preserve_words_levels_views_and_skip_overwrites():
    calls = []
    words = np.array([[0x7fa12345, 0x80000000, 0x3f800001]], np.uint32).view(np.float32)
    def load():
        calls.append('cold')
        return words.copy(order='F')
    with FrameworkPhysics() as native:
        native.put_deferred_field('state', 'lazy', words.shape, words.dtype, load, nlevels=2)
        assert calls == []
        replacement = np.ones_like(words)
        native.put_field('state', 'lazy', replacement, level=1, nlevels=2)
        assert calls == []
        replacement[:] = 9
        assert np.all(native.get_field('state', 'lazy', level=1) == 1)
        assert native.get_field('state', 'lazy', level=2).tobytes() == words.tobytes()
        assert calls == ['cold']
        view = native.field_storage('state', 'lazy', level=2)
        native.put_field('state', 'lazy', np.full_like(words, 3), level=2, nlevels=2)
        assert np.all(view == 3)
        assert native._storage['state', 'lazy'].loader is None
        native.put_deferred_field('state', 'zeroed', words.shape, words.dtype, load, nlevels=2)
        native.put_zero_field('state', 'zeroed', words.shape, words.dtype, nlevels=2)
        assert calls == ['cold']
        assert np.all(native.get_field('state', 'zeroed') == 0)


def test_moving_zero_mappings_keep_private_time_levels_and_live_views(monkeypatch):
    import os, mmap
    if os.name != 'posix':
        pytest.skip('anonymous private mappings are the Linux moving path')
    monkeypatch.setenv('WOOF_HEX_MOVING_RESIDENT', '1')
    shape = (55, 1024)
    with FrameworkPhysics() as native:
        native.put_zero_field('state', 'zero', shape, np.float32, nlevels=2)
        one = native.field_storage('state', 'zero')
        two = native.field_storage('state', 'zero', level=2)
        assert isinstance(one.base, mmap.mmap) and isinstance(two.base, mmap.mmap)
        assert one.flags.f_contiguous and not np.any(one.view(np.uint32))
        one[:] = 7
        assert not np.any(two.view(np.uint32))
        native.put_zero_field('state', 'zero', shape, np.float32, nlevels=2)
        assert not np.any(one.view(np.uint32))
    one[0, 0] = 3  # An outstanding NumPy view owns its mapping after the session closes.
    assert one[0, 0] == 3


@pytest.mark.parametrize('copy_on_write',[False,True])
def test_hydrate_exact_words_and_separate_time_levels(tmp_path,copy_on_write):
    # Rust transport layout includes its dummy column and arbitrary bit words.
    words = np.array([[0,0x80000000,0x7fc12345],
                      [0x3f800001,0x7f800000,0xff800000]], np.uint32, order='F')
    value = words.view(np.float32)
    raw = value.tobytes(order='F')
    (tmp_path/'words.bin').write_bytes(raw)
    row = dict(pool='state',name='tracer',pool_name='tracer',kind='real',
               dimensions=['nVertLevels','nCells'],shape=[2,3],nlevels=2)
    record = {k:row[k] for k in ('pool','name','kind','dimensions','shape')}
    record.update(payload='words.bin',sha256=hashlib.sha256(raw).hexdigest())
    (tmp_path/'manifest.json').write_text(json.dumps(
        dict(schema='mpas-native-pool-payload-v1',fields=[record])))
    layout = SimpleNamespace(fields=[row],configs={'config_dt':90.125},
        dimensions={'nVertLevels':2,'nCells':2},indices={})
    with FrameworkPhysics() as framework:
        hydration = NativePoolHydration(framework,layout,tmp_path)
        hydration.copy_on_write = copy_on_write
        hydration.populate()
        for level in (1,2):
            assert framework.get_field('state','tracer',level=level).tobytes(order='F') == raw
        changed = value.copy(order='F')
        changed[0,0] = np.float32(4)
        framework.put_field('state','tracer',changed,level=2,nlevels=2)
        assert framework.get_field('state','tracer',level=1).tobytes(order='F') == raw
        # A scheme's mutable level-one storage must never mutate its cold
        # payload, nor the words returned by a later hydration read.
        framework.field_storage('state','tracer',level=1)[0,0] = np.float32(9)
        assert hydration.field_buffer('state','tracer').tobytes(order='F') == raw
        assert (tmp_path/'words.bin').read_bytes() == raw
        assert framework.get_real_config('config_dt').tobytes() == np.float32(90.125).tobytes()
        with pytest.raises(ValueError,match='dtype, shape'):
            framework.put_field('state','tracer',np.zeros((3,2),np.float32),nlevels=2)
    with pytest.raises(RuntimeError,match='closed'):
        framework.get_field('state','tracer')


def test_registration_halo_layers_and_dummy_storage():
    with FrameworkPhysics() as framework:
        framework.decomposition('nCells',global_ids=np.array([3,7,3,7],np.int32),
                                owned=2,halo_ends=[3,4])
        value = np.array([[1,2,-1,-2,99],[3,4,-3,-4,100]],np.float32,order='F')
        framework.put_field('state','rho',value)
        framework.register_field('state','rho',dimensions=('nVertLevels','nCells'),
                                 decomposed=True)
        framework.exchange_group('outer',('rho',),halo_layers=(2,))
        framework.exchange_halo_group('outer')
        expected = value.copy(); expected[:,3] = value[:,1]
        np.testing.assert_array_equal(framework.get_field('state','rho'),expected)
        framework.exchange_halo('rho')
        expected[:,2] = value[:,0]
        np.testing.assert_array_equal(framework.get_field('state','rho'),expected)
        with pytest.raises(ValueError,match='no owned source'):
            framework.decomposition('nEdges',global_ids=np.array([1,2],np.int32),owned=1)


@pytest.mark.parametrize('start,dt,expected,doy,leap',[
    ('2023-12-31_23:59:30',30,'2024-01-01_00:00:00',1,True),
    ('2024-02-28_23:59:59',1,'2024-02-29_00:00:00',60,True),
    ('2000-02-29_23:59:00',60,'2000-03-01_00:00:00',61,True),
    ('2100-02-28_23:59:59',1,'2100-03-01_00:00:00',60,False),
])
def test_clock_boundaries(start,dt,expected,doy,leap):
    clock = PhysicsClock(start=start,stop='2101-01-01_00:00:00',dt=dt)
    clock.advance()
    assert clock.current_time.stamp() == expected
    assert clock.current_time.day_of_year == doy
    words = clock.calendar_words()
    assert words['julian'].tobytes() == np.float32(doy-1).tobytes()
    assert words['leapyear'] is leap


def test_noleap_and_360_day_arithmetic():
    assert ModelTime.parse('2024-02-28_23:59:59','gregorian_noleap').add(1).stamp() == '2024-03-01_00:00:00'
    assert ModelTime.parse('2024-02-30_23:59:59','360_day').add(1).stamp() == '2024-03-01_00:00:00'
    assert ModelTime.parse('2024-01-01_00:00:00').add(Fraction(-1,2)).seconds == Fraction(172799,2)


def test_calendar_flags_constants_module_dims_and_state_time():
    with FrameworkPhysics() as framework:
        framework.clock(start='2024-02-28_23:59:30',stop='2024-03-01_00:00:00',dt=30)
        framework.put_text_field('state','initial_time','2024-02-28_12:00:00',nlevels=2)
        for level in (1,2):
            framework.put_field('state','Time',np.array(43170,np.float32),level=level,nlevels=2)
        framework.initialize_state_time()
        assert framework.get_text_field_storage('state','xtime').tobytes() == b'2024-02-28_23:59:30'.ljust(512,b' ')
        assert framework.get_text_field('state','initial_time',level=2) == '2024-02-28_12:00:00'
        assert framework.get_field('state','Time') == np.float32(43170)
        framework.advance_clock()
        framework.update_state_time(30,clock_advanced=True)
        assert framework.get_text_field('state','xtime',level=2) == '2024-02-29_00:00:00'
        assert framework.get_field('state','Time',level=2) == np.float32(43200)
        words = framework.calendar_state(StepFlags(cam_abs=True,cam_restart=False,radiation_lw=True))
        assert words['julian'] == np.float32(59)
        assert framework.get_variable('doabsems')[0] == 1
        assert framework.get_variable('l_camlw')[0] == 0
        assert framework.get_variable('l_radtlw')[0] == 1
        framework.mpas_atm_set_dims(nvertlevels_val=4,maxedges_val=6,maxedges2_val=12,num_scalars_val=8)
        assert framework._core_dimensions == (4,6,12,8)
        assert framework.get_constant('r_d').view(np.uint32) == np.float32(287).view(np.uint32)
        for name,row in framework.abi['constants'].items():
            assert framework.get_constant(name).view(np.uint32) == row['word']


def test_manager_initialization_allocation_and_finalize():
    text = '&physics\nconfig_physics_suite="convection_permitting"\n/'
    config = resolve_namelist(text)
    layout = RegistryLayout(registry_source(),config,text,
        dict(nCells=2,nEdges=3,nVertices=2,maxEdges=6,maxEdges2=12,
             vertexDegree=3,nVertLevels=4))
    with FrameworkPhysics(layout=layout) as framework:
        framework.clock(start='2024-02-29_12:15:30',stop='2024-03-01_00:00:00',dt=180)
        initialization = FrameworkInitialization(SimpleNamespace(native=framework,layout=layout,config=config))
        initialization.physics_run_init(None)
        for name,value in dict(ims=1,ime=2,kme=5,num_soils=4,n_microp=2,dt_microp=90,n_cu=1,has_reqc=0).items():
            assert framework.get_variable(name)[0] == value
        framework.allocate_forall_physics()
        assert framework.get_variable('t_p')[0].shape == (2,5,1)
        assert framework.get_variable('t_p')[1] == (1,1,1)
        framework.scheme_state = {'test':object()}
        initialization.atmphys_finalize(None)
        assert not framework.scheme_state


def test_default_route_never_requests_native_resources(tmp_path,monkeypatch):
    from woof.hex.mpasphys import forecast_v842,resources_build
    reader = tmp_path/'reader'; reader.touch()
    calls = []
    def installed(keys,**kw):
        calls.append(keys)
        assert keys == ('pool-reader',)
        return {'pool-reader':reader}
    monkeypatch.setattr(resources_build,'ensure_route_resources',installed)
    for key in ('MPAS_NATIVE_LIBRARY','MPAS_NATIVE_POOL_READER','MPAS_SOURCE'):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(ValueError,match='requires --init'):
        forecast_v842.run_forecast(SimpleNamespace(),None,'')
    assert calls == [('pool-reader',)]


def test_replay_captured_native_clock_and_geometry_words():
    from woof.hex.mpasphys.framework_init_v842 import initialize_directions
    captured = json.loads(Path(__file__).with_name('framework_native_words_v842.json').read_text())
    geometry = captured['geometry']
    lat = np.array(geometry['latitude'],np.uint32).view(np.float32)
    lon = np.array(geometry['longitude'],np.uint32).view(np.float32)
    east,north = initialize_directions(lat,lon)
    np.testing.assert_array_equal(east.view(np.uint32),geometry['east'])
    np.testing.assert_array_equal(north.view(np.uint32),geometry['north'])
    for case in captured['calendar']:
        clock = PhysicsClock(start=case['start'],stop='2101-01-01_00:00:00',
                             dt=case['dt'],calendar=case['calendar'])
        for row in case['rows']:
            words = clock.calendar_words(); t = clock.current_time
            assert words['current_date'] == row['stamp']
            assert [t.year,t.month,t.day,t.day_of_year,int(t.leapyear)] == row['dates']
            assert int(words['julian'].view(np.uint32)) == row['julian_word']
            clock.advance()


@pytest.mark.parametrize('binding', ['framework', 'radshared'])
def test_ozone_formatted_records_keep_rkind_and_native_loop_order(tmp_path, monkeypatch, binding):
    import ctypes
    from woof.hex.mpasphys.framework_resources_v842 import decode_ozone_tables
    from woof.hex.mpasphys.radshared_v842 import load_ozone_resources
    decode = decode_ozone_tables if binding == 'framework' else load_ozone_resources
    def forbidden_decoder(*args, **kwargs):
        raise AssertionError('ozone table binding must not load a decoder library')
    monkeypatch.setattr(ctypes, 'CDLL', forbidden_decoder)
    records = {
        'OZONE_PLEV.TBL':['1.000000059604644775390625', '-0.0'],
        'OZONE_LAT.TBL':[str(i-32) for i in range(64)],
        'OZONE_DAT.TBL':[str(i)+'D-4' for i in range(12*64*2)],
    }
    hashes = {}
    for name, words in records.items():
        # A Fortran list-directed READ consumes one record, even if it has
        # further columns. Ozone's loops read one value per READ statement.
        data = ''.join(word+' 999\n' for word in words).encode()
        (tmp_path/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    resource = dict(table_directory=str(tmp_path),table_sha256=hashes,nOznLevels=2,
                    rust_decoder=dict(path=str(tmp_path/'missing-decoder.so'),sha256='0'*64))
    actual = decode(resource,np)
    assert actual['pin'].view(np.uint32).tolist() == [0x3f800000,0x80000000]
    assert actual['lat_ozone'].shape == (64,)
    assert actual['ozmixin'].shape == (12,64,2)
    assert actual['ozmixin'][11,63,1] == np.float32('0.1535')
    (tmp_path/'OZONE_LAT.TBL').write_text('tampered')
    with pytest.raises(ValueError,match='SHA256|sha256|differs'):
        decode(resource,np)


@pytest.mark.parametrize('classification,folded,expected', [
    ('MODIFIED_IGBP_MODIS_NOAH', 2, [17, 1, 17, 21]),
    ('USGS', 0, [21, 1, 21, 21]),
])
def test_lake_fold_uses_owned_pool_words_and_preserves_dummy(classification, folded, expected):
    from woof.hex.mpasphys.runtime_v842 import fold_lake_class
    fields = [dict(pool='sfc_input', name=name, pool_name=storage)
              for name, storage in [('ivgtyp', 'ivgtyp'), ('lu_index', 'lu_index'),
                                    ('iswater_lu', 'iswater'), ('mminlu', 'mminlu')]]
    layout = SimpleNamespace(fields=fields, dimensions={'nCells': 3})
    with FrameworkPhysics() as framework:
        framework.put_text_field('sfc_input', 'mminlu', classification)
        framework.put_field('sfc_input', 'iswater', np.array(17, np.int32))
        for name, dtype in [('ivgtyp', np.int32), ('lu_index', np.float32)]:
            framework.put_field('sfc_input', name, np.array([21, 1, 21, 21], dtype))
        assert fold_lake_class(framework, layout) == folded
        for name, dtype in [('ivgtyp', np.int32), ('lu_index', np.float32)]:
            assert framework.get_field('sfc_input', name).tobytes() == np.array(expected, dtype).tobytes()


def test_framework_session_binds_surface_service_before_resident_admission(tmp_path):
    """Manager monthly updates cannot depend on a particular resident suite."""
    from woof.hex.mpasphys.session_v842 import NativePhysicsSession
    text = '&physics\nconfig_physics_suite="none"\n/'
    config = resolve_namelist(text)
    assert str(config.service_backends['surface_update']) == 'native-cpu'
    layout = RegistryLayout(registry_source(), config, text,
        dict(nCells=2, nEdges=3, nVertices=2, maxEdges=6, maxEdges2=12,
             vertexDegree=3, nVertLevels=4))
    layout.fields = [row for row in layout.fields
                     if row['name'] in ('indexToCellID', 'indexToEdgeID', 'indexToVertexID')]
    records = []
    for row in layout.fields:
        words = np.arange(1, row['shape'][0] + 1, dtype=np.int32)
        data = words.tobytes(order='F')
        filename = row['name'] + '.bin'
        (tmp_path / filename).write_bytes(data)
        record = {key: row[key] for key in ('pool', 'name', 'kind', 'dimensions', 'shape')}
        record.update(payload=filename, sha256=hashlib.sha256(data).hexdigest())
        records.append(record)
    (tmp_path/'manifest.json').write_text(json.dumps(
        dict(schema='mpas-native-pool-payload-v1', fields=records)))
    # No active scheme uses the device in this construction-only CPU test.
    # Real pools, registration, clock, hooks and surface service are retained.
    with FrameworkPhysics() as framework:
        session = NativePhysicsSession(framework, layout, tmp_path, config,
            dt=1200, start='2023-03-31_12:00:00', stop='2023-03-31_15:00:00',
            stream_intervals={}, post_transport=lambda call: None, cuda_device=np)
        assert not hasattr(session, 'resident_glue')
        assert session.surface_service.session is session
        for operation in session.surface_service.OPERATIONS:
            assert session.hooks.external[operation] == session.surface_service.invoke
        assert framework.lib is None


@pytest.mark.parametrize('enabled', [False, True])
def test_framework_ozone_hook_does_not_require_resident_admission(enabled):
    text = '&physics\nconfig_physics_suite="none"\n/'
    config = resolve_namelist(text)
    session = SimpleNamespace(native=None, config=config,
        layout=SimpleNamespace(dimensions={}, configs={'config_o3climatology': enabled}))
    operations = FrameworkInitialization(session).operations()
    assert ('o3climatology_from_MPAS' in operations) is enabled


@pytest.mark.parametrize('resident', [False, True])
def test_framework_ozone_dispatch_preserves_calendar_words_and_pool_authority(monkeypatch, resident):
    """Exercise the real framework hook and pool transaction, not CUDA arithmetic."""
    from woof.hex.mpasphys import Backend, SlotCall
    from woof.hex.mpasphys import radshared_v842 as ozone
    from woof.hex.mpasphys.device_pools_v842 import NativeDevicePoolViews
    from woof.hex.mpasphys.driver_v842 import NativeDriverHooks, NativePoolArguments

    class Device(np.ndarray):
        def get(self, order='C'):
            return np.array(self, copy=True, order=order)

    device = SimpleNamespace(ndarray=Device, float32=np.float32, int32=np.int32,
        uint32=np.uint32, asarray=lambda value: np.array(value, copy=True).view(Device),
        ascontiguousarray=lambda value: np.ascontiguousarray(value).copy().view(Device))
    rows = [dict(pool=pool, name=name, pool_name=name, dimensions=dimensions)
            for pool, name, dimensions in (
                ('atm_input', 'ozmixm', ['nMonths', 'nOznLevels', 'nCells']),
                ('diag_physics', 'o3clim', ['nOznLevels', 'nCells']))]
    layout = SimpleNamespace(dimensions={'nCells': 3, 'nCellsSolve': 2}, fields=rows,
        pools={'atm_input', 'diag_physics'}, configs={'config_o3climatology': True})
    sentinel = np.array([0x80000000, 0x7fc12345, 0x3f800001, 0xff800000], np.uint32)
    monthly = np.resize(sentinel, (12, 2, 4)).view(np.float32)
    original = np.resize(sentinel, (2, 4)).view(np.float32)
    results = [np.array([[0x80000000, 0x7fc54321], [0x40000001, 0x3f800000]], np.uint32),
               np.array([[0x40400001, 0x80000000], [0x7fc45678, 0x40800001]], np.uint32)]
    seen = []

    with FrameworkPhysics() as framework:
        framework.clock(start='2024-02-28_23:59:30', stop='2024-03-01_00:00:00', dt=60)
        framework.put_field('atm_input', 'ozmixm', monthly)
        framework.put_field('diag_physics', 'o3clim', original)
        arguments = NativePoolArguments(framework,
            pools={name: name for name in ('mesh', 'atm_input', 'diag_physics')}, its=1, ite=2,
                                        synchronize=lambda *args: None)
        config = resolve_namelist('&physics\nconfig_physics_suite="none"\n/')
        session = SimpleNamespace(native=framework, layout=layout, config=config, arguments=arguments)
        provider = session.device_pools = NativeDevicePoolViews(session, device)
        provider.resident = resident

        def interpolation_boundary(call):
            pools = call.pools
            seen.append((pools['julian'].tobytes(), pools['current_date']))
            assert pools['nCellsSolve'] == 2
            assert pools['atm_input']['ozmixm'].tobytes() == monthly[..., :2].tobytes()
            target = pools['diag_physics']['o3clim']
            target.view(np.uint32)[...] = results[len(seen)-1]
            return target

        monkeypatch.setattr(ozone, 'ozone_slot', interpolation_boundary)
        hooks = NativeDriverHooks(framework, arguments, after_call=lambda *args: None,
            external=FrameworkInitialization(session).operations(),
            save_fields=lambda: {}, restore_fields=lambda value: None)
        for step in range(2):
            flags = StepFlags(radiation_lw=True)
            calendar = framework.calendar_state(flags)
            arguments.update_calendar(calendar)
            call = SlotCall('o3climatology_from_MPAS', 'driver', Backend.HEX_CUDA,
                            step+1, step*60., 60., flags, {}, config.execution_options)
            output = hooks.invoke('o3climatology_from_MPAS', call)
            np.testing.assert_array_equal(output.view(np.uint32), results[step])
            expected = original.copy()
            if not resident:
                expected.view(np.uint32)[:, :2] = results[step]
            np.testing.assert_array_equal(framework.get_field('diag_physics', 'o3clim').view(np.uint32),
                                          expected.view(np.uint32))
            assert framework.get_field('atm_input', 'ozmixm').tobytes() == monthly.tobytes()
            assert seen[-1] == (calendar['julian'].tobytes(), calendar['current_date'])
            if resident:
                assert provider.resident_buffers['diag_physics', 'o3clim', 1][1] is output
            framework.advance_clock()
        assert seen[0][1].startswith('2024-02-28_23:59:30')
        assert seen[1][1].startswith('2024-02-29_00:00:30')
        assert seen[0][0] != seen[1][0]
        assert framework.lib is None
