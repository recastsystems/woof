"""A storm-following move continues in the resident forecast process (woof.hex.mpasphys.moving_resident_v842).

THE COST THIS REMOVES (HEX-SHAPES-1CARD 2026-10-29, 48 -> 3 km storm-following region on a path, 306,507 cells): a
move cost 194 s from the stop to the next region's first step, of which the carrier pickle (14 s out, ~15 s back),
the next process's pool decode, mesh and remap geometry and its cold CUDA start were most.  These tests pin the
contract the controller and the forecast process share: the carrier handed over in memory comes back with the
words and array layout a sealed file gives, only a named move continues in the process, the continuation applies
the next segment's own environment, and a failed continuation leaves the carrier on disk for the recovery.
"""
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import moving_run
from woof.hex.mpasphys import moving_resident_v842 as resident


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(resident, '_R', resident._Resident())
    monkeypatch.setenv(resident.RESIDENT_ENV, '1')
    # next_segment sets the next segment's run-state variables in os.environ itself (as the process must); put
    # them back, or a later test's moving route reads this test's carrier path (WOOF_HEX_MOVING_CHECKPOINT_IN).
    saved = {name: os.environ.get(name) for name in resident.SEGMENT_VARIABLES}
    yield
    resident._R.watch_stop.set()
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _payload():
    fortran = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(4, 6))
    base = np.arange(40, dtype=np.float64).reshape(5, 8)
    return dict(schema='gpuwm-hex.moving-boundary-mpas/v1', time_seconds=1800.0, valid_time='2023-03-31_12:30:00',
                state=dict(rho=np.ones((3, 5), np.float32), view=base[:, ::2], fortran=fortran),
                physics=dict(pools={('sfc', 'tsk', 1): base[1]}))


def test_class_inputs_are_claimed_before_the_following_prefetch_replaces_them(tmp_path,monkeypatch):
    layout=object()
    loaded,classes=object(),[dict(rate=1,marker='current')]
    monkeypatch.setattr(resident,'inputs',lambda *args:loaded)
    key=(id(layout),str(tmp_path.resolve()))
    resident._R.prepared['lts-classes']=dict(key=key,value=classes)
    actual,claimed=resident.class_inputs(None,layout,tmp_path,False,lambda:None)
    # Admission runs after this return. A following region can now publish
    # its own preparation without making the current segment wait or lose it.
    future=[dict(rate=2,marker='following')]
    resident._R.prepared['lts-classes']=dict(key=('following',),value=future)
    class FutureThread:
        def is_alive(self):return True
        def join(self):raise AssertionError('current segment must not wait for following preparation')
    resident._R.prep_thread=FutureThread()
    assert actual is loaded and claimed is classes
    assert claimed[0]['marker']=='current'
    assert resident._R.prepared['lts-classes']['value'] is future
    resident._R.prep_thread=None


def test_the_carrier_in_memory_has_the_sealed_files_words_and_layout(tmp_path):
    payload = _payload()
    spec = dict(index=1, member='mn1')
    receipt = resident.store_carrier(tmp_path / 'boundary.pkl', payload, dict(physics_fields=1), spec)
    assert receipt['resident'] and not (tmp_path / 'boundary.pkl').exists()
    written = json.loads((tmp_path / 'boundary.pkl.json').read_text())
    assert written['time_seconds'] == 1800.0 and written['resident'] is True
    carried = resident.take_carrier(tmp_path / 'boundary.pkl')
    from_file = pickle.loads(pickle.dumps(payload, protocol=5))
    for name in ('rho', 'view', 'fortran'):
        a, b = carried['state'][name], from_file['state'][name]
        assert a.dtype == b.dtype and a.shape == b.shape and a.strides == b.strides
        assert a.tobytes(order='A') == b.tobytes(order='A')
    assert carried['state']['fortran'].flags.f_contiguous
    assert resident.take_carrier(tmp_path / 'boundary.pkl') is None   # taken once


def test_load_carrier_reads_the_handed_over_carrier_first(tmp_path):
    from woof.hex.mpasphys.moving_v842 import load_carrier
    resident.store_carrier(tmp_path / 'boundary.pkl', _payload(), {}, dict(index=1, member='mn1'))
    carried = load_carrier(tmp_path / 'boundary.pkl')
    assert carried['valid_time'] == '2023-03-31_12:30:00'


def _control(tmp_path, target):
    control = SimpleNamespace(directory=tmp_path, stopped=None if target is None else {'target': target})
    return control


def test_only_a_move_the_controller_named_hands_off(tmp_path):
    spec = dict(schema=resident.SPEC_SCHEMA, member='mn1', index=1, argv=['forecast'], segment=str(tmp_path))
    (tmp_path / resident.NEXT_NAME).write_text(json.dumps(spec))
    assert resident.hands_off(_control(tmp_path, 'mn1'))['index'] == 1
    assert resident.hands_off(_control(tmp_path, 'mn2')) is None          # the record names another member
    assert resident.hands_off(_control(tmp_path, None)) is None           # a run end, not a move
    os.environ.pop(resident.RESIDENT_ENV)
    assert resident.hands_off(_control(tmp_path, 'mn1')) is None          # --no-resident


def test_the_continuation_takes_the_next_segments_environment_and_closes_the_last(tmp_path, monkeypatch):
    old = tmp_path / 'segment-000'
    new = tmp_path / 'segment-001'
    (old / 'control').mkdir(parents=True)
    (new / 'control').mkdir(parents=True)
    monkeypatch.setenv('WOOF_HEX_MOVING_CONTROL', str(old / 'control'))
    monkeypatch.setenv('WOOF_HEX_MOVING_OPERATOR', 'stale')
    monkeypatch.setattr(os, 'dup2', lambda *a: None)
    # This is the continuation/control contract, including on the CPU test tier.
    # CuPy may be installed while devices are hidden; the real release owns CUDA.
    monkeypatch.setattr(resident, 'release_device_memory', lambda: None)
    spec = dict(schema=resident.SPEC_SCHEMA, member='mn1', index=1, argv=['forecast', '--out', 'x'],
                segment=str(new), env={'WOOF_HEX_MOVING_CONTROL': str(new / 'control'),
                                       'WOOF_HEX_MOVING_CHECKPOINT_IN': str(old / 'boundary.pkl')})
    resident._R.continuation = spec
    argv = resident.next_segment()
    assert argv == ['forecast', '--out', 'x']
    assert json.loads((old / resident.CLOSED_NAME).read_text())['next_index'] == 1
    assert os.environ['WOOF_HEX_MOVING_CONTROL'] == str(new / 'control')
    assert 'WOOF_HEX_MOVING_OPERATOR' not in os.environ    # a fresh process would not have inherited it
    assert resident.next_segment() is None
    resident._R.watch_stop.set()


def test_a_failed_continuation_seals_the_carrier_for_recovery(tmp_path):
    resident.store_carrier(tmp_path / 'boundary.pkl', _payload(), dict(physics_fields=1), dict(index=1, member='m'))
    resident.take_carrier(tmp_path / 'boundary.pkl')        # the continued segment read it, then failed
    assert resident.seal_handed('test') == [str((tmp_path / 'boundary.pkl').resolve())]
    from woof.hex.mpasphys.moving_v842 import load_carrier
    assert load_carrier(tmp_path / 'boundary.pkl')['time_seconds'] == 1800.0
    assert moving_run.carrier_sealed(tmp_path / 'boundary.pkl')


def test_the_controller_accepts_a_resident_carrier_receipt(tmp_path):
    boundary = tmp_path / 'boundary.pkl'
    assert not moving_run.carrier_sealed(boundary)
    boundary.with_suffix('.pkl.json').write_text(json.dumps(dict(resident=True, time_seconds=1800.0)))
    assert moving_run.carrier_sealed(boundary)
    boundary.with_suffix('.pkl.json').write_text(json.dumps(dict(time_seconds=1800.0)))
    assert not moving_run.carrier_sealed(boundary)          # a file receipt without its file is not a carrier


def test_a_prepared_piece_is_taken_only_for_its_own_inputs():
    resident._R.prepared['inputs'] = dict(key=('a',), value=1)
    assert resident._take('inputs', ('b',)) is None
    assert 'inputs' not in resident._R.prepared
    resident._R.prepared['inputs'] = dict(key=('a',), value=1)
    assert resident._take('inputs', ('a',)) == 1


def test_a_finished_segments_kernel_entries_are_dropped_and_the_default_caches_kept(monkeypatch):
    """forget_segment_kernels drops (cache, name) entries of per-class caches, keeps each module's own _CACHE."""
    import sys
    import types
    from woof.hex.cuda_backend import KernelCache
    default, finished = object.__new__(KernelCache), object.__new__(KernelCache)
    module = types.ModuleType('woof.hex._resident_test_kernels')
    module._CACHE = default
    module._KERNELS = {(default, 'a'): 1, (finished, 'a'): 2, (finished, 'b', (('n', 4),)): 3, ('plain', 'c'): 4}
    winds = types.ModuleType('woof.hex.cuda_dynamics_v841')
    winds._REFERENCE_WIND = {(1, 2, 3): ('arrays',)}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, winds.__name__, winds)
    assert resident.forget_segment_kernels() == 3
    assert module._KERNELS == {(default, 'a'): 1, ('plain', 'c'): 4}
    assert winds._REFERENCE_WIND == {}


@pytest.mark.parametrize('changed', ['rate', 'cell_l2g', 'cell_ring', 'n_owned_cells', '_global_cells'])
def test_prepared_classes_reject_changed_admission_and_keep_an_unchanged_class(changed):
    from copy import deepcopy
    from woof.hex.mpasphys.moving_lts_prepare_v842 import matching
    partition = SimpleNamespace(cell_l2g=np.array([0, 2, 1], np.int32),
                                cell_ring=np.array([0, 0, 1], np.int16),
                                n_owned_cells=2, _global_cells=3)
    unit = SimpleNamespace(rate=2, layout=partition)
    other = SimpleNamespace(rate=1, layout=deepcopy(partition))
    predicted = [dict(rate=u.rate, partition=deepcopy(u.layout)) for u in (unit, other)]
    if changed == 'rate':
        unit.rate = 4
    elif isinstance(getattr(partition, changed), np.ndarray):
        getattr(partition, changed)[0] += 1
    else:
        setattr(partition, changed, getattr(partition, changed) + 1)
    assert matching(predicted, [unit, other]) == [None, predicted[1]]


@pytest.mark.parametrize('changed', ['ownership','ordering','complete','extent','rate','unknown'])
def test_prepared_class_inputs_survive_only_an_ownership_change(changed):
    from copy import deepcopy
    from woof.hex.mpasphys.moving_lts_prepare_v842 import matching
    from woof.hex.mpasphys.lts_v842 import session_partition
    before=SimpleNamespace(cell_l2g=np.array([0,2,1],np.int32),edge_l2g=np.array([1,0],np.int32),
        vertex_l2g=np.array([0,1],np.int32),cell_ring=np.array([0,0,1],np.int16),
        n_owned_cells=2,n_owned_edges=1,n_owned_vertices=1,
        cell_row_complete=np.ones(3,bool),edge_row_complete=np.ones(2,bool),vertex_row_complete=np.ones(2,bool),
        _global_cells=3,_global_edges=2,_global_vertices=2)
    after=deepcopy(before)
    after.n_owned_cells=1
    after.cell_ring[:]=[0,1,1]
    rate=2
    if changed=='ordering':after.cell_l2g[:]=after.cell_l2g[::-1]
    if changed=='complete':after.cell_row_complete[0]=False
    if changed=='extent':after._global_cells+=1
    if changed=='rate':rate=4
    if changed=='unknown':after.unknown=True
    mesh=SimpleNamespace(partition_layout=before,partition_layout_v841=before)
    record=dict(rate=2,partition=before,inputs=(mesh,),gpu_driver=object(),physics_backend=object())
    result=matching([record],[SimpleNamespace(rate=rate,layout=after)],input_only=True)
    if changed!='ownership':
        assert result==[None]
        return
    assert result==[record] and record['partition'] is after
    assert mesh.partition_layout is after
    for name,value in vars(session_partition(before)).items():
        np.testing.assert_array_equal(value,getattr(session_partition(after),name))
    assert after.n_owned_cells==1
    assert record['gpu_preparation']['reused_for_ownership_change'] is True


def test_prepared_driver_takes_every_carried_state_and_diagnostic_word():
    from woof.hex.mpasphys.moving_class_gpu_v842 import restore_driver
    class Array:
        def __init__(self): self.value = None
        def set(self, value): self.value = value.copy()
    class Driver:
        @classmethod
        def from_host(cls, *args, **kwargs): raise AssertionError('must reuse prepared driver')
    state_names = ('rho','rho_theta','rho_u','rho_w','scalars')
    saved_names = ('theta_m','exner','density_perturbation','rho_theta_perturbation',
                   'pressure_perturbation','normal_velocity','vertical_velocity')
    state = SimpleNamespace(**{n: np.array([i, -0.], np.float32) for i,n in enumerate(state_names)},
                            time_seconds=1800.)
    saved = SimpleNamespace(**{n: np.array([i+10, -0.], np.float32) for i,n in enumerate(saved_names)})
    driver = Driver()
    driver.config = 'same'
    driver.atmosphere = SimpleNamespace(state=SimpleNamespace(**{n:Array() for n in state_names}),
                                       saved=SimpleNamespace(**{n:Array() for n in saved_names}))
    assert restore_driver({'gpu_driver':driver}, Driver.from_host, None, state, None, None, 'same',
                          saved_diagnostics=saved) is driver
    assert driver.atmosphere.state.time_seconds == 1800.
    for names, host, device in ((state_names,state,driver.atmosphere.state),
                                (saved_names,saved,driver.atmosphere.saved)):
        for n in names: assert getattr(device,n).value.tobytes() == getattr(host,n).tobytes()


def test_pending_driver_kernel_cache_survives_only_until_its_segment_uses_it(monkeypatch):
    import sys, types
    from woof.hex.cuda_backend import KernelCache
    pending, finished = object.__new__(KernelCache), object.__new__(KernelCache)
    module = types.ModuleType('woof.hex._resident_gpu_pending_test')
    module._KERNELS = {(pending,'a'):1,(finished,'a'):2}
    monkeypatch.setitem(sys.modules,module.__name__,module)
    resident._R.prepared['lts-classes'] = dict(value=[dict(gpu_driver=SimpleNamespace(cache=pending))])
    resident.forget_segment_kernels()
    assert module._KERNELS == {(pending,'a'):1}
    resident._R.prepared.clear()
    resident.forget_segment_kernels()
    assert module._KERNELS == {}


@pytest.mark.parametrize('configured,expected', [(-1.,12345.),(3000.,3000.)])
def test_ahead_hydration_resolves_runtime_length_after_output_config_reset(tmp_path,configured,expected):
    from woof.hex.mpasphys.pool_v842 import NativePoolHydration
    from woof.hex.mpasphys.session_v842 import NativePhysicsSession
    class AtClock(Exception): pass
    class Native:
        def __init__(self): self.values={}; self.nominal=None
        def config(self,name,value): self.values[name]=value
        def put_field(self,pool,name,value): self.nominal=np.asarray(value).copy()
        def clock(self,**kwargs): raise AtClock()
    native=Native()
    layout=SimpleNamespace(fields=[],configs=dict(config_len_disp=configured,
                                                  config_calendar_type='gregorian'))
    hydration=object.__new__(NativePoolHydration)
    hydration.native=native;hydration.layout=layout;hydration.payload=tmp_path
    hydration.records={('mesh','nominalMinDc'): {}}
    hydration.field_buffer=lambda *args:np.array(12345.,np.float32)
    output_snapshot=dict(layout.configs)
    with pytest.raises(AtClock):
        NativePhysicsSession(native,layout,tmp_path,SimpleNamespace(),dt=18.75,
            start='2023-03-31_12:00:00',stop='2023-03-31_12:30:00',
            stream_intervals={},post_transport=lambda **kwargs:None,prehydrated=hydration)
    assert output_snapshot['config_len_disp']==configured
    assert native.values['config_len_disp']==expected
    assert float(native.nominal)==expected
    assert layout.configs['config_len_disp']==expected


def test_prepared_dycore_config_keeps_the_pre_hydration_namelist(monkeypatch):
    import sys
    from woof.hex.mpasphys import moving_class_gpu_v842 as ahead, pool_v842, route_table
    from woof.hex import cuda_backend, cuda_physics_prep_v841, class_work_arena
    class Native:
        def __init__(self,**kwargs): pass
    class Hydration:
        def __init__(self,native,layout,payload): self.layout=layout
        def populate(self): self.layout.configs['config_len_disp']=3000.
    class Driver:
        @classmethod
        def from_host(cls,*args,**kwargs): return SimpleNamespace(config=args[4])
    def spec(config,layout,dt,*,configs=None):
        selected=dict(layout.configs if configs is None else configs)
        return Driver,lambda **kwargs:None,SimpleNamespace(config_les_model='none',
            config_mix_scalars=False,config_len_disp=selected['config_len_disp'])
    monkeypatch.setattr(ahead,'driver_spec',spec)
    monkeypatch.setattr(route_table,'framework_entry',lambda *args:Native)
    monkeypatch.setattr(pool_v842,'NativePoolHydration',Hydration)
    monkeypatch.setattr(cuda_backend,'require_cuda',lambda **kwargs:None)
    monkeypatch.setattr(cuda_physics_prep_v841.CudaMpasToPhysGeometryV841,'from_host',lambda *args:None)
    monkeypatch.setattr(class_work_arena,'_ARENAS',[])
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(
        get_default_memory_pool=lambda:SimpleNamespace(used_bytes=lambda:0)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    item=dict(layout=SimpleNamespace(configs=dict(config_len_disp=-1.)),payload='unused')
    inputs=(None,SimpleNamespace(vertical_grid=None,terrain_metrics=None),None,None,None,None,None)
    ahead.prepare(item,inputs,SimpleNamespace(),SimpleNamespace(),18.75)
    assert item['layout'].configs['config_len_disp']==3000.
    assert item['gpu_driver'].config.config_len_disp==-1.
    # Actual admitted geometry can be built before any local pool payload exists.
    actual = dict(layout=SimpleNamespace(configs=dict(config_len_disp=-1.)))
    ahead.prepare_driver(actual,inputs,SimpleNamespace(),SimpleNamespace(),18.75)
    assert actual['gpu_driver'].config.config_len_disp == -1.
    assert 'physics_backend' not in actual and 'physics_hydration' not in actual


def test_device_geometry_receipt_keeps_all_prepared_inputs(tmp_path, monkeypatch, capsys):
    from woof.hex import physics_suites, moving_vertical_cuda
    from woof.hex.mpasphys import runtime_v842, pool_v842, moving_v842, moving_lts_prepare_v842
    args = SimpleNamespace(scratch=tmp_path, out=tmp_path, grid=tmp_path/'grid', init=tmp_path/'init')
    layout = SimpleNamespace(configs={})
    mesh = SimpleNamespace(arrays={'areaCell':np.ones(2)})
    vertical = SimpleNamespace(zgrid=np.ones((3, 2)))
    loaded = (mesh, SimpleNamespace(vertical_grid=vertical))
    weights = SimpleNamespace(backend='cuda')
    resident._R.current = dict(mesh=mesh, vertical=vertical, decode=dict(
        sources=[], config=SimpleNamespace(table_key='test'), reader='reader', source='source',
        text='', initial_state=None))
    monkeypatch.setattr(resident, '_parse', lambda argv:args)
    monkeypatch.setattr(resident, '_decode_key', lambda call:'key')
    monkeypatch.setattr(pool_v842, 'decode_registry_pools', lambda *a, **kw:layout)
    monkeypatch.setattr(physics_suites, 'load_table', lambda:{'test':{}})
    monkeypatch.setattr(runtime_v842, 'keeps_initialized_deformation', lambda row:False)
    monkeypatch.setattr(runtime_v842, 'load_native_dynamics_inputs', lambda *a, **kw:loaded)
    monkeypatch.setattr(moving_v842, 'build_weights', lambda **kw:weights)
    monkeypatch.setattr(moving_v842, 'largest_donors', lambda w:None)
    monkeypatch.setattr(moving_vertical_cuda, 'prepare_geometry', lambda w:dict(prepared=True, live_bytes=1))
    from woof.hex import moving_momentum_cuda
    from woof.hex import moving_kernel_prep
    monkeypatch.setattr(moving_kernel_prep, 'prepare', lambda w:[])
    monkeypatch.setattr(moving_momentum_cuda, 'prepare', lambda *args:None)
    monkeypatch.setattr(moving_lts_prepare_v842, 'predict', lambda *a, **kw:None)
    resident.prepare(dict(argv=[], index=1, member='next', env=dict(
        WOOF_HEX_MOVING_OPERATOR=str(tmp_path/'operator'), WOOF_HEX_MOVING_SOURCE_GRID=str(tmp_path/'old'))))
    assert resident._R.prepared['weights']['value'][0] is weights
    assert set(resident._R.prepared) == {'decode','inputs','weights'}
    assert 'remap-device-geometry' in capsys.readouterr().out
    assert any(row['stage']=='remap-device-geometry' for row in resident._R.prep_rows)


def test_gpu_preparation_budget_counts_shared_slotted_state_once():
    from dataclasses import dataclass
    from woof.hex.mpasphys.moving_class_gpu_v842 import _array_bytes
    @dataclass(slots=True)
    class State:
        rho: object
        saved: object
    value = np.ones((55,19), np.float32)
    assert _array_bytes(State(value,value)) == value.nbytes


def test_gpu_driver_budget_excludes_global_mesh_parent_and_shared_host_aliases():
    from woof.hex.mpasphys.moving_class_gpu_v842 import _driver_input_bytes
    local = np.ones((55,19),np.float32)
    parent = np.ones((55,1900),np.float32)
    mesh = SimpleNamespace(arrays={'local':local}, partition_source_mesh=SimpleNamespace(arrays={'all':parent}))
    grid = SimpleNamespace(vertical_grid=SimpleNamespace(zz=local), terrain_metrics=SimpleNamespace(zb_cell=local))
    assert _driver_input_bytes((mesh,grid,None,SimpleNamespace(rho=local),None,None,('qv',))) == local.nbytes
