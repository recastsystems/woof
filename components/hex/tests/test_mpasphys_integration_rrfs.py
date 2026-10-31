"""Actual fork ABI closure and scratch seam routing checks; no forecast claim."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest
import site_support
from woof.hex.mpasphys.rrfs.driver_rrfs import (
    Backend,NativeDriverHooks,NativePoolArguments,SlotCall,SlotRegistry,StepFlags,
    package_selection,resolve_namelist,specification)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def fork_native():
    path=os.environ.get('RRFS_NATIVE_LIBRARY');loader=os.environ.get('RRFS_NATIVE_LOADER')
    if not path or not loader:pytest.skip('explicit actual r2 artifact and loader required')
    return load_module('rrfs_r2_native_artifact',Path(loader)).NativePhysics,Path(path)


@pytest.mark.parametrize('suite',['hrrrv5','rrfs_mesoscale_reference','rrfs_convection_permitting','rrfs_none'])
def test_actual_r2_abi_closes_fork_driver(fork_native,suite):
    kind,path=fork_native
    config=resolve_namelist(f"&physics config_physics_suite='{suite}' /")
    with kind(path,test_only=True,pin_set='rrfs') as native:
        args=NativePoolArguments(native,pools={},its=1,ite=1,synchronize=lambda entry,call:None)
        hooks=NativeDriverHooks(native,args,after_call=lambda entry,call:None,
            external={'physics_timetracker_state':lambda call:None,'post_transport_prepare':lambda call:None},
            save_fields=lambda:{},restore_fields=lambda fields:None)
        assert hooks.missing_operations(config)==()
        registry=SlotRegistry();hooks.register_slots(registry,config);registry.require(config)
        # This writes real native typed configuration pools, without pretending
        # that a hydrated mesh or scientific initialization was supplied.
        hooks.configure(config)
        assert native.build['source_pin']==specification()['model_commit']


def test_smoke_package_rules_include_rwc_without_inventing_driver_activation():
    config=resolve_namelist("&physics config_physics_suite='none' /\n&chemistry config_rwc_scheme='on' /")
    packages=package_selection(config)
    assert packages['mpas_smoke_fine'] and packages['mpas_anthro_in'] and packages['mpas_rwc_in']
    assert not config.active('smoke') # native physics_manager excludes RWC from this predicate
    assert not packages['mpas_smoke_in'] and not packages['mpas_dust_in']


def test_broken_callback_is_selected_at_first_native_step():
    from test_mpasphys_driver_rrfs import driver,deck
    config=resolve_namelist(deck('hrrrv5'),backends={'pbl':'hex-cuda'})
    instance=driver(config)
    key=('pbl',config.selectors['pbl'],Backend.HEX_CUDA)
    def broken(call):raise RuntimeError(f'broken rublten at step {call.itimestep}')
    instance.registry._callbacks[key]=broken
    with pytest.raises(RuntimeError,match='broken rublten at step 1'):
        instance.begin_step(itimestep=1,time_seconds=0,pools={})
    assert instance.calls[-1]['slot']=='pbl' and instance.calls[-1]['backend']=='hex-cuda'


@pytest.fixture(scope='module')
def proposal():
    path=os.environ.get('RRFS_INTEGRATOR_FORECAST_PROPOSAL')
    if not path:pytest.skip('explicit scratch proposal required')
    return load_module('woof.hex.mpasphys.rrfs_forecast_proposal',Path(path))


@pytest.mark.parametrize('suite',['hrrrv5','rrfs_mesoscale_reference','rrfs_convection_permitting','rrfs_none'])
def test_table_proposal_routes_without_suite_branches(proposal,suite):
    from types import SimpleNamespace
    config,text=proposal.native_route(SimpleNamespace(physics_namelist=None,physics_suite=suite,
        physics=['config_tempo_hailaware=.false.',"config_dust_scheme='off'"]))
    assert config.pin_set=='rrfs'
    assert not config.options['config_tempo_hailaware']
    assert config.options['config_dust_scheme']=='off'
    assert '&physics_mp_tempo' in text and '&chemistry' in text


@pytest.mark.parametrize('suite',['mesoscale_reference','convection_permitting','none'])
def test_proposal_keeps_unqualified_mpas_routes(proposal,suite):
    from types import SimpleNamespace
    from woof.hex.mpasphys.driver_v842 import resolve_namelist as resolve_old
    config,text=proposal.native_route(SimpleNamespace(physics_namelist=None,physics_suite=suite,physics=None))
    assert config.receipt()==resolve_old(text).receipt()


@pytest.mark.parametrize('shape',[(0,),(0,2),(2,0),(2,0,3)])
@pytest.mark.parametrize('dtype',['float32','int32'])
def test_real_native_abi_preserves_empty_registry_fields(fork_native,shape,dtype):
    import numpy as np
    kind,path=fork_native
    with kind(path,test_only=True,pin_set='rrfs') as native:
        native.pool('diag_physics')
        fields=np.empty(shape,dtype=dtype,order='F')
        for level in [1,2]:
            native.put_field('diag_physics','empty',fields,level=level,nlevels=2)
            result=native.get_field('diag_physics','empty',level=level)
            assert result.shape==shape and result.dtype==fields.dtype and result.size==0


@pytest.mark.parametrize('native_suite,public_alias',[
    ('mesoscale_reference','rrfs_mesoscale_reference'),
    ('convection_permitting','rrfs_convection_permitting'),('none','rrfs_none')])
def test_explicit_table_alias_selects_unchanged_native_namelist(proposal,tmp_path,native_suite,public_alias):
    from types import SimpleNamespace
    path=tmp_path/'namelist.atmosphere'
    original=f"&physics config_physics_suite='{native_suite}' /"
    path.write_text(original)
    config,text=proposal.native_route(SimpleNamespace(physics_namelist=path,
        physics_suite=public_alias,physics=None))
    assert config.pin_set=='rrfs' and config.options['config_physics_suite']==native_suite
    assert text==original


def test_stream_metadata_preserves_native_duplicate_attribute_semantics(fork_native,monkeypatch):
    from rrfs_native_stream_tree import stream_tree
    import xml.etree.ElementTree as ET
    kind,path=fork_native
    monkeypatch.setattr('woof.hex.mpasphys.native_backend.NativePhysics',kind)
    decks=sorted(site_support.rrfs_reference('decks','RRFS_PUBLISHED_DECK_ROOT',vendor='rrfs-decks').glob('*/*/streams.atmosphere'))
    if not decks:pytest.skip('r0 decks not published')
    checked=0
    def metadata(element):
        return element.tag,element.attrib,tuple(metadata(c) for c in element)
    for deck in decks:
        native=stream_tree(deck,path)
        try:
            strict=ET.parse(deck).getroot()
        except ET.ParseError as error:
            assert 'duplicate attribute' in str(error)
            lbc=[e for e in native if e.get('name')=='lbc_in']
            assert len(lbc)==1 and lbc[0].get('filename_interval')==lbc[0].get('input_interval')
        else:
            # Defaults are native metadata, absent from the source XML. Explicit
            # aliases resolve through the pristine C extractor.
            for resolved, declared in zip(native, strict, strict=True):
                resolved.attrib.pop('filename_interval', None)
                declared.attrib.pop('filename_interval', None)
            assert metadata(native)==metadata(strict),str(deck)
        checked+=1
    assert checked==len(decks)


@pytest.mark.parametrize('flags', range(8))
def test_fork_reset_adapter_matches_complete_native_routine(fork_native, flags):
    import ctypes
    import numpy as np
    from woof.hex.mpasphys.rrfs.driver_rrfs import reset_native_diagnostics
    oracle = os.environ.get('RRFS_ALARM_REFERENCE_DIR')
    if not oracle:
        pytest.skip('complete native reset oracle required')
    initial = np.array([[0x80000000, 1, 0x7f800000],
                        [0x3f800000, 0xbf800000, 0x7fc00000],
                        [2, 0x80000001, 0x41200000]], dtype=np.uint32, order='F')
    expected = initial.view(np.float32).copy(order='F')
    lib = ctypes.CDLL(str(Path(oracle) / 'libreset.so'))
    lib.reset_fields.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.reset_fields(3, flags, expected.ctypes.data)
    names = ('refl10cm_1km_max', 'max_hail_diameter_sfc_acc',
             'max_hail_diameter_column_acc')
    kind, path = fork_native
    with kind(path, test_only=True, pin_set='rrfs') as native:
        native.pool('diag_physics')
        for i, name in enumerate(names):
            if flags & (1 << i):
                native.put_field('diag_physics', name, initial[:, i].view(np.float32))
        native.put_field('diag_physics', 'unrelated', initial[:, 0].view(np.float32))
        reset_native_diagnostics(native)
        for i, name in enumerate(names):
            if flags & (1 << i):
                actual = native.get_field('diag_physics', name)
                assert np.array_equal(actual.view(np.uint32), expected[:, i].view(np.uint32))
            else:
                assert ('diag_physics', name) not in native._fields
        assert np.array_equal(native.get_field('diag_physics', 'unrelated').view(np.uint32),
                              initial[:, 0])


def test_second_ring_adapter_retains_native_order_and_dummy_storage(fork_native, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import numpy as np
    from woof.hex.mpasphys.rrfs.driver_rrfs import initialize_native_core
    from woof.hex.mpasphys.session_v842 import NativePhysicsSession
    provider = ModuleType('woof.hex.core_rrfs')
    counts = np.array([2, 0, 1], dtype=np.int32)
    neighbors = np.array([[2, 1, -1], [-1, -1, -1], [0, -1, -1]], dtype=np.int32)
    calls = []
    def bridge(mesh, *, cell_solve_count):
        calls.append((mesh, cell_solve_count))
        return counts, neighbors
    provider.initialize_second_ring_rrfs = bridge
    monkeypatch.setitem(sys.modules, 'woof.hex.core_rrfs', provider)
    kind, path = fork_native
    with kind(path, test_only=True, pin_set='rrfs') as native:
        native.pool('mesh')
        native.put_field('mesh', 'nCellsOnCellsOnCell', np.array([11, 12, 13, 14], np.int32))
        native.put_field('mesh', 'cellsOnCellsOnCell',
                         np.arange(21, 33, dtype=np.int32).reshape(3, 4, order='F'))
        session = NativePhysicsSession.__new__(NativePhysicsSession)
        session.native = native
        session.layout = SimpleNamespace(dimensions={'nCellsSolve': 2})
        mesh = object()
        initialize_native_core(session, mesh)
        assert calls == [(mesh, 2)]
        assert np.array_equal(native.get_field('mesh', 'nCellsOnCellsOnCell'), [2, 0, 13, 14])
        assert np.array_equal(native.get_field('mesh', 'cellsOnCellsOnCell'),
                              [[3, 24, 27, 30], [2, 25, 28, 31], [23, 26, 29, 32]])


@pytest.mark.parametrize('suite', ['hrrrv5', 'mesoscale_reference', 'convection_permitting'])
def test_complete_hydrated_native_session_selects_fork_provider(fork_native, suite):
    """Exercise the whole proposed session with actual Rust-decoded mesh pools.

    This binds native storage, clock, driver, hooks and every active slot. It
    executes no scientific initialization or timestep and proves no forecast.
    The external trial tree is explicit because the shared consumer is owned
    by the integrator; this test must not silently patch production imports.
    """
    trial = os.environ.get('RRFS_SESSION_CLOSURE_ROOT')
    if not trial:
        pytest.skip('explicit complete decoded-pool/session proposal fixture required')
    trial = Path(trial)
    pool_provider = load_module('woof.hex.mpasphys.rrfs_pool_session_trial',
        trial / 'tree/src/hexcore/mpasphys/pool_v842.py')
    session_provider = load_module('woof.hex.mpasphys.rrfs_session_trial',
        trial / 'tree/src/hexcore/mpasphys/session_v842.py')
    text = (trial / (suite + '-namelist.atmosphere')).read_text()
    config = resolve_namelist(text)
    layout = pool_provider.RegistryLayout(trial / 'reference', config, text,
        json.loads((trial / (suite + '-dimensions.json')).read_text()))
    kind, library = fork_native
    with kind(library, test_only=True, pin_set='rrfs') as native:
        session = session_provider.NativePhysicsSession(native, layout,
            trial / ('pools-' + suite) / 'native-pools', config,
            dt=layout.configs['config_dt'], start=layout.configs['config_start_time'],
            stop='2023-04-01_12:00:00', stream_intervals={},
            post_transport=lambda call: None,
            core_operations={'atm_reset_diagnostics': lambda call: None})
        from woof.hex.mpasphys.rrfs.driver_rrfs import PhysicsDriver, PhysicsManager
        assert type(session.driver) is PhysicsDriver
        assert type(session.driver.manager) is PhysicsManager
        assert session.hooks.missing_operations(config) == ()
        session.driver.registry.require(config)
        assert native.pin_set == 'rrfs'
        assert not session._initialized
        for struct in layout.root.findall('./var_struct'):
            if not layout.enabled(struct):
                continue
            pool = struct.get('name_in_code', struct.get('name'))
            for array in struct.findall('var_array'):
                if not layout.enabled(array):
                    assert native._dimension_values[pool, 'num_' + array.get('name')] == 0
        for field in layout.fields:
            key = (field['pool'], field['pool_name'])
            if field['kind'] == 'character':
                assert key in native._text_fields
            else:
                assert key in native._fields
                _, shape, levels = native._fields[key]
                assert shape == tuple(field['shape'])
                assert levels == field['nlevels']


def test_whole_fork_table_selection_keeps_extensionless_inputs_and_identical_aliases(tmp_path):
    from woof.hex.mpasphys.rrfs.driver_rrfs import native_table_inputs
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir(); second.mkdir()
    for name in specification()['native_table_names']:
        (first / name).write_bytes(b'opaque native input')
    (first / 'SOILPARM.TBL').write_bytes(b'native land table')
    (second / 'SOILPARM.TBL').write_bytes(b'native land table')
    (second / 'RRTMG_SW_DATA.DBL').write_bytes(b'native radiation table')
    (first / 'PUBLICATION.json').write_text('{}')
    (first / 'native.stdout.log').write_text('not a native input')
    selected = native_table_inputs([first, second, first])
    assert len(selected) == len(specification()['native_table_names']) + 2
    assert {p.name for p in selected} == set(specification()['native_table_names']) | {
        'SOILPARM.TBL', 'RRTMG_SW_DATA.DBL'}
    assert next(p for p in selected if p.name == 'SOILPARM.TBL') == first / 'SOILPARM.TBL'


def test_whole_fork_table_selection_refuses_conflicting_inputs_before_installation(tmp_path):
    from woof.hex.mpasphys.rrfs.driver_rrfs import native_table_inputs
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir(); second.mkdir()
    (first / 'VEGPARM.TBL').write_bytes(b'first native table')
    (second / 'VEGPARM.TBL').write_bytes(b'different native table')
    with pytest.raises(ValueError, match='conflicting native table bytes for VEGPARM.TBL'):
        native_table_inputs([first, second])


def test_native_table_selection_leaves_missing_inputs_to_pristine_initialization(tmp_path):
    from woof.hex.mpasphys.rrfs.driver_rrfs import native_table_inputs
    assert native_table_inputs([tmp_path]) == ()


def test_explicit_native_table_roots_match_ordered_native_forecast_staging(tmp_path):
    from woof.hex.mpasphys.rrfs.driver_rrfs import native_table_inputs
    base, first, last = (tmp_path / name for name in ('base', 'first', 'last'))
    for directory in (base, first, last):
        directory.mkdir()
    for name in specification()['native_table_names']:
        (base / name).write_bytes(b'original native bytes')
    (base / 'RRTMG_SW_DATA').write_bytes(b'single precision radiation')
    (base / 'RRTMG_SW_DATA.DBL').write_bytes(b'double precision radiation')
    (first / 'ugwp_limb_tau.nc').write_bytes(b'first explicit variant')
    (last / 'ugwp_limb_tau.nc').write_bytes(b'last explicit variant')
    (last / 'qr_acr_qg_data_tempo_v3').write_bytes(b'explicit tempo variant')
    (last / 'PUBLICATION.json').write_text('{}')
    selected = {p.name: p for p in native_table_inputs(
        [base], override_roots=[first, last, last])}
    assert selected['ugwp_limb_tau.nc'] == last / 'ugwp_limb_tau.nc'
    assert selected['qr_acr_qg_data_tempo_v3'] == last / 'qr_acr_qg_data_tempo_v3'
    assert selected['freeze_water_data_tempo_v3'] == base / 'freeze_water_data_tempo_v3'
    assert selected['RRTMG_SW_DATA'].read_bytes() == b'single precision radiation'
    assert selected['RRTMG_SW_DATA.DBL'].read_bytes() == b'double precision radiation'
    assert 'PUBLICATION.json' not in selected
    assert specification()['native_table_override_keyword'] == 'override_roots'


def test_explicit_table_roots_do_not_hide_ambiguous_built_inputs(tmp_path):
    from woof.hex.mpasphys.rrfs.driver_rrfs import native_table_inputs
    roots = [tmp_path / name for name in ('base', 'ambiguous', 'override')]
    for index, directory in enumerate(roots):
        directory.mkdir()
        (directory / 'SOILPARM.TBL').write_bytes(bytes([index]))
    with pytest.raises(ValueError, match='conflicting native table bytes for SOILPARM.TBL'):
        native_table_inputs(roots[:2], override_roots=roots[2:])


@pytest.mark.parametrize('suite', ['hrrrv5', 'mesoscale_reference', 'convection_permitting'])
def test_inactive_constituent_metadata_preserves_active_fields_and_input_dimensions(suite):
    from types import SimpleNamespace
    import xml.etree.ElementTree as ET
    from woof.hex.mpasphys.rrfs.driver_rrfs import prepare_native_layout
    config = resolve_namelist(f"&physics config_physics_suite='{suite}' /")
    root = ET.fromstring('''<registry><var_struct name="emission_input">
      <var_array name="e_bb_in" packages="inactive"><var name="smoke" array_group="smoke"/></var_array>
      <var_array name="active" packages="active"><var name="a"/></var_array>
      </var_struct></registry>''')
    fields = [{'pool': 'emission_input', 'name': 'active'}]
    dimensions = {'nlcat': 20, 'nscat': 16}
    layout = SimpleNamespace(root=root, fields=fields, dimensions=dimensions,
                             indices={('emission_input', 'num_active'): 1},
                             enabled=lambda node: node.get('packages') != 'inactive')
    prepare_native_layout(layout)
    assert layout.indices == {('emission_input', 'num_active'): 1,
        ('emission_input', 'num_e_bb_in'): 0, ('emission_input', 'index_smoke'): -1,
        ('emission_input', 'smoke_start'): -1, ('emission_input', 'smoke_end'): -1}
    assert layout.fields is fields
    assert layout.dimensions == dimensions
    assert config.native_layout_prepare.endswith(':prepare_native_layout')


def test_native_registry_cleanup_joins_carriers_after_attribute_inheritance():
    import xml.etree.ElementTree as ET
    from woof.hex.mpasphys.rrfs.driver_rrfs import prepare_native_registry
    root = ET.fromstring('''<registry>
      <var_struct name="state" time_levs="2">
        <var_array name="scalars" type="real" dimensions="nCells Time">
          <var name="qv" packages="moist"/>
        </var_array>
      </var_struct>
      <var_struct name="state" time_levs="1" packages="smoke">
        <var_array name="scalars" type="real" dimensions="nCells Time">
          <var name="smoke"/>
          <var name="explicit" packages="dust"/>
        </var_array>
        <var_struct name="nested"><var name="a"/></var_struct>
      </var_struct>
      <var_struct name="state"><var_struct name="nested"><var name="b"/></var_struct></var_struct>
      <streams><stream name="input" type="input"><var_array name="scalars"/></stream></streams>
      <streams><stream name="input" type="output"><var name="smoke"/></stream>
               <stream name="emissions" type="input"><var name="explicit"/></stream></streams>
    </registry>''')
    prepare_native_registry(root)
    structs = root.findall('var_struct')
    assert len(structs) == 1 and structs[0].get('time_levs') == '2'
    arrays = structs[0].findall('var_array')
    assert len(arrays) == 1 and arrays[0].get('time_levs') == '2'
    assert [(v.get('name'), v.get('packages')) for v in arrays[0]] == [
        ('qv', 'moist'), ('smoke', 'smoke'), ('explicit', 'dust')]
    nested = structs[0].findall('var_struct')
    assert len(nested) == 1 and [v.get('name') for v in nested[0]] == ['a', 'b']
    assert nested[0][0].get('packages') == 'smoke'
    assert nested[0][0].get('time_levs') == '1'
    # Native pushes attributes once, before merging: the later unqualified
    # member must not acquire the first declaration's smoke package.
    assert nested[0][1].get('packages') is None
    assert nested[0][1].get('time_levs') is None
    assert len(root.findall('streams')) == 1
    streams = root.findall('./streams/stream')
    assert [s.get('name') for s in streams] == ['input', 'emissions']
    assert streams[0].get('type') == 'input'
    assert [v.get('name') for v in streams[0]] == ['scalars', 'smoke']
    assert resolve_namelist("&physics config_physics_suite='hrrrv5' /").native_registry_prepare.endswith(
        ':prepare_native_registry')


def test_native_climatology_stream_uses_selected_table_bytes_and_real_option(tmp_path):
    from woof.hex.mpasphys.rrfs.driver_rrfs import native_table_stream_inputs
    base, override = tmp_path / 'base', tmp_path / 'override'
    base.mkdir(); override.mkdir()
    row, = specification()['native_table_streams']
    (base / row['filename']).write_bytes(b'native base')
    (override / row['filename']).write_bytes(b'native selected')
    off = resolve_namelist("&physics config_physics_suite='none' /")
    on = resolve_namelist("&physics config_physics_suite='none', config_ngw_scheme=.true. /")
    assert native_table_stream_inputs(off, [base], override_roots=[override]) == {}
    assert native_table_stream_inputs(on, [base], override_roots=[override]) == {
        row['stream']: override / row['filename']}
    assert native_table_stream_inputs(on, [tmp_path]) == {}


def test_ported_route_reads_table_streams_from_the_fetched_fork_tree_without_mpas_source(tmp_path, monkeypatch):
    """hrrrv5 with no --mpas-source took the wheel's Registry-only tree as its table root and refused at start."""
    from woof.hex.mpasphys import forecast_v842
    from woof.hex.mpasphys.route_table import framework_entry
    from woof.hex.mpasphys.rrfs import framework_rrfs
    fetched, named = tmp_path / 'fetched', tmp_path / 'named'
    monkeypatch.setattr(framework_rrfs, 'table_source', lambda: fetched)
    config = resolve_namelist("&physics config_physics_suite='hrrrv5' /")
    registry = framework_entry(config, 'registry_source')()
    assert not (registry / 'src/core_atmosphere/physics/physics_wrf/files').exists()
    assert forecast_v842.native_table_source(config, None, registry, True) == fetched
    assert forecast_v842.native_table_source(config, named, named, True) == named
    assert forecast_v842.native_table_source(config, None, named, False) == named


def test_native_initializer_keeps_original_internal_namelist_groups(tmp_path):
    import hashlib
    from woof.hex.mpasphys.rrfs.driver_rrfs import stage_native_initialization_namelist
    original = tmp_path / 'original'
    contents = b"&physics config_physics_suite='none' /\r\n&nssl_mp_params cnor=1.e6 /\r\n"
    original.write_bytes(contents)
    work = tmp_path / 'work'; work.mkdir()
    assert stage_native_initialization_namelist(original, work) == hashlib.sha256(contents).hexdigest()
    assert (work / 'namelist.atmosphere').read_bytes() == contents
    stage_native_initialization_namelist(original, work)
    (work / 'namelist.atmosphere').write_bytes(b'different native options')
    with pytest.raises(ValueError, match='native initialization namelist already differs'):
        stage_native_initialization_namelist(original, work)
    assert (work / 'namelist.atmosphere').read_bytes() == b'different native options'


@pytest.mark.parametrize('suite', ['hrrrv5', 'rrfs_mesoscale_reference', 'rrfs_convection_permitting'])
def test_landed_door_mode_controls_keep_fork_exact_authority(tmp_path, suite):
    from types import SimpleNamespace
    from woof.hex.mpasphys.forecast_v842 import native_route
    original = f"&physics config_physics_suite='{suite}' /"
    path = tmp_path / 'namelist.atmosphere'
    path.write_text(original)
    args = SimpleNamespace(physics_namelist=path, physics_suite=None, physics=None,
                           physics_mode='exact', physics_correct_skip=None)
    exact, text = native_route(args, default_mode='correct')
    # The door's default is the row's table default (hex-cuda: every slot on its GPU port where this install has one).
    before = resolve_namelist(original, default_backend=specification()['default_backend'])

    def routed(config):
        # The route names every active slot that runs on the CPU, each with why it has no GPU port here.
        receipt = dict(config.receipt())
        cpu = receipt.pop('cpu_slots', [])
        assert [row['slot'] for row in cpu] == [s for s in config.backends
                                                if config.active(s) and str(config.backends[s]) == 'native-cpu']
        assert all(row['reason'] and 'default MPAS physics backend is native-cpu' not in row['reason'] for row in cpu)
        return receipt
    assert routed(exact) == routed(before)
    assert text == original and exact.mode == 'exact' and not exact.corrections
    # Native receives no Hex orchestration controls, whether provided in the
    # namelist or through the real door arguments.
    path.write_text(original + "\n&nhyd_model config_physics_mode='exact' /")
    nml, text = native_route(args, default_mode='correct')
    assert routed(nml) == routed(before)
    assert 'config_physics_mode' not in text
    args.physics_mode = None
    path.write_text(original)
    correct, text = native_route(args, default_mode='correct')
    assert correct.options == before.options and correct.backends == before.backends
    assert correct.selectors == before.selectors and not correct.corrections
    gap = correct.receipt()['correct_mode']['gaps'][0]
    assert gap['finding'] == specification()['correct_mode_inventory']['finding']
    assert 'fork' in gap['reason'] and 'v8.3.1/GSL' in gap['breakage']
    assert 'CORRECT MODE: named gaps' in correct.gap_warning()
    # Applying v842's RP6 machinery to this fork would change its call schedule.
    assert not correct.correct('RP6') and not correct.correct('RP4')


@pytest.mark.parametrize('kind,direction,inp,out,explicit,timelevels,expected', [
    ('stream','output',None,'01:00:00',None,None,'01:00:00'),
    ('immutable_stream','input;output','initial_only','01:00:00',None,None,'01:00:00'),
    ('stream','input;output','00:30:00','01:00:00',None,None,'00:30:00'),
    ('stream','input','initial_only',None,None,None,'none'),
    ('stream','output',None,'final_only',None,'2','output'),
    ('immutable_stream','output',None,'final_only',None,'2','none'),
    ('stream','output',None,'01:00:00','none',None,'none'),
    ('stream','output',None,'01:00:00','output_interval',None,'01:00:00'),
    ('stream','input','initial_only',None,'input_interval',None,'none'),
])
def test_pristine_stream_metadata_resolves_filename_policy(fork_native, tmp_path, kind,
        direction, inp, out, explicit, timelevels, expected):
    from rrfs_native_stream_tree import stream_tree
    import xml.etree.ElementTree as ET
    _, library = fork_native
    tree = ET.Element('streams')
    attrs = {'name':'probe','type':direction,'filename_template':'field.$h.nc'}
    for name, value in [('input_interval',inp),('output_interval',out),
                        ('filename_interval',explicit),('output_timelevels',timelevels)]:
        if value is not None: attrs[name] = value
    ET.SubElement(tree, kind, attrs)
    path = tmp_path / 'streams.xml'
    ET.ElementTree(tree).write(path)
    assert stream_tree(path, library)[0].get('filename_interval') == expected


def test_stream_filename_alias_uses_pristine_native_extractor(fork_native, tmp_path):
    from rrfs_native_stream_tree import stream_tree
    _, library = fork_native
    path = tmp_path / 'streams.xml'
    path.write_text("<streams><stream name='cadence' type='output' filename_template='x.$h.nc' "
                    "output_interval='00:30:00'/><stream name='probe' type='output' "
                    "filename_template='y.$h.nc' output_interval='stream:cadence:output_interval'/></streams>")
    assert stream_tree(path, library)[1].get('filename_interval') == '00:30:00'


def test_fork_registry_accepts_landed_empty_correction_metadata():
    config = resolve_namelist("&physics config_physics_suite='hrrrv5' /",
                              backends={'microphysics':'hex-cuda'})
    registry = SlotRegistry()
    callback = lambda call: None
    registry.register('microphysics', config.selectors['microphysics'], 'hex-cuda',
                      callback, corrections=())
    assert registry._callbacks['microphysics', 'mp_tempo', Backend.HEX_CUDA] is callback
    assert registry.declared('microphysics','mp_tempo',Backend.HEX_CUDA) == frozenset()
    with pytest.raises(ValueError, match='fork source and replay qualification'):
        SlotRegistry().register('microphysics','mp_tempo','hex-cuda',callback,corrections=('RP6',))



def test_fork_reset_zeroes_the_resident_device_maxima():
    """mpas_atm_core.F:883-894,971-1002 (fork): on a resident run the GPU copy is the authority, so the reset after a
    diagnostics write zeroes it too (it zeroed only host words, and the maxima ran from the start of the run)."""
    import numpy as np
    from types import SimpleNamespace
    from woof.hex.mpasphys.rrfs.driver_rrfs import reset_native_diagnostics
    names = ('refl10cm_1km_max', 'max_hail_diameter_sfc_acc', 'max_hail_diameter_column_acc')

    class Host:
        def __init__(self):
            self._fields = {('diag_physics', n): (np.float32, (4,), 1) for n in names}
            self.words = {n: np.full(4, 7.0, np.float32) for n in names}

        def put_field(self, pool, name, value, *, nlevels=1):
            self.words[name] = np.array(value)

    class Views:
        resident = True

        def __init__(self):
            self.device = {n: np.full(3, 9.0, np.float32) for n in names}  # owned extent, one dummy column short

        def field(self, pool, name):
            return self.device[name]

    host, views = Host(), Views()
    reset_native_diagnostics(host, session=SimpleNamespace(device_pools=views))
    for n in names:
        assert not host.words[n].any() and not views.device[n].any()
    # A non-resident session (or none) keeps the host-only reset.
    host, views = Host(), Views()
    views.resident = False
    reset_native_diagnostics(host, session=SimpleNamespace(device_pools=views))
    for n in names:
        assert not host.words[n].any() and views.device[n].all()


def test_ngw_door_binds_installed_climatology_without_a_false_deck_warning(tmp_path, capsys):
    from woof.hex.mpasphys.forecast_v842 import deck_stream_inputs, require_table_stream_inputs
    from woof.hex.mpasphys.rrfs.driver_rrfs import native_table_stream_inputs
    row, = specification()['native_table_streams']
    tables = tmp_path / 'fork' / 'src/core_atmosphere/physics/physics_wrf/files'
    tables.mkdir(parents=True)
    climatology = tables / row['filename']
    climatology.write_bytes(b'climatology binding only, no field decoding')
    deck = tmp_path / 'streams.atmosphere'
    deck.write_text(f'<streams><immutable_stream name="{row["stream"]}" type="input" '
                    f'filename_template="{row["filename"]}" input_interval="initial_only" /></streams>')
    on = resolve_namelist("&physics config_physics_suite='none', config_ngw_scheme=.true. /")
    selected = native_table_stream_inputs(on, [tables])
    assert selected == {row['stream']: climatology}
    assert deck_stream_inputs(deck, tmp_path / 'init.nc', fallback=selected) == {}
    assert capsys.readouterr().err == ''
    local = tmp_path / row['filename']
    local.write_bytes(b'case-specific native climatology')
    assert deck_stream_inputs(deck, tmp_path / 'init.nc', fallback=selected) == {row['stream']: local}
    require_table_stream_inputs(on, selected)
    with pytest.raises(FileNotFoundError, match='active climatology stream ugwp_ngw_in requires ugwp_limb_tau.nc'):
        require_table_stream_inputs(on, {})
    # A user-supplied stream file is sufficient even without installed tables.
    require_table_stream_inputs(on, {row['stream']: climatology})
    off = resolve_namelist("&physics config_physics_suite='none' /")
    require_table_stream_inputs(off, {})
