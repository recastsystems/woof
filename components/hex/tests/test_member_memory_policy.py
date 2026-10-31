"""The shared-card policy preserves release allocations for single members."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof.hex.device_memory_release import DevicePhaseRelease, shared_card_memory


def test_policy_validation_and_release_defaults():
    assert not shared_card_memory({})
    assert not shared_card_memory({'WOOF_HEX_MEMBER_MEMORY': 'single'})
    assert shared_card_memory({'WOOF_HEX_MEMBER_MEMORY': 'shared-card'})
    with pytest.raises(ValueError, match='must be single or shared-card'):
        shared_card_memory({'WOOF_HEX_MEMBER_MEMORY': 'shared'})
    solo = DevicePhaseRelease(environ={})
    shared = DevicePhaseRelease(environ={'WOOF_HEX_MEMBER_MEMORY': 'shared-card'})
    assert (solo.pool_min_bytes, solo.headroom) == (8 << 30, .15)
    assert (shared.pool_min_bytes, shared.headroom) == (1 << 30, 1.)
    override = DevicePhaseRelease(environ={'WOOF_HEX_MEMBER_MEMORY': 'shared-card',
        'GPUWM_HEX_POOL_RELEASE_MIN_BYTES': '123', 'GPUWM_HEX_RELEASE_HEADROOM': '.25'})
    assert (override.pool_min_bytes, override.headroom) == (123, .25)


@pytest.mark.parametrize('shared', [False, True])
def test_mynn_worker_capacity_and_scratch_lifetime(shared, monkeypatch):
    from woof.hex.mpasphys import bl_mynn_v842 as m
    monkeypatch.setattr(m, '_SHARED_WORKSPACE', {})
    module = m.MynnCudaModule.__new__(m.MynnCudaModule)
    module.cp = SimpleNamespace(empty=np.empty, uint8=np.uint8, int32=np.int32,
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(ptr=0),
            runtime=SimpleNamespace(getDevice=lambda: 0,
                getDeviceProperties=lambda d: {'multiProcessorCount': 2},
                memGetInfo=lambda: (1 << 30, 2 << 30))))
    module._resident = module._workers = None
    module._shared_card_memory = shared
    first, e1, workers = module._storage(1024, 16)
    second, e2, next_workers = module._storage(1024, 16)
    assert workers == next_workers == (128 if shared else 512)
    assert (first is second) is (not shared)
    assert e1 is not e2


@pytest.mark.parametrize('mode,expected', [('single', 1536), ('shared-card', 512)])
def test_thompson_budget_and_explicit_override(mode, expected, monkeypatch):
    from woof.hex.mpasphys.thompson_v842 import workspace_budget_bytes
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', mode)
    monkeypatch.delenv('WOOF_HEX_THOMPSON_WORKSPACE_MIB', raising=False)
    assert workspace_budget_bytes() == expected << 20
    monkeypatch.setenv('WOOF_HEX_THOMPSON_WORKSPACE_MIB', '768')
    assert workspace_budget_bytes() == 768 << 20


def test_single_snapshot_keeps_output_inputs(monkeypatch):
    from woof.hex.mpasphys.resident_output_diagnostics_v842 import ResidentOutputDiagnostics
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'single')
    obj = ResidentOutputDiagnostics.__new__(ResidentOutputDiagnostics)
    obj.convective = SimpleNamespace(reset=lambda: None)
    obj.acoustic = {'old': True}
    obj.buffers = {('state', 'scalars'): object()}
    before = dict(obj.buffers)
    obj.after_snapshot()
    assert obj.buffers == before and obj.acoustic == {}


def test_single_shortwave_keeps_release_width(monkeypatch):
    from woof.hex.mpasphys.rrtmg_sw_v842 import MPASShortwave
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'single')
    obj = MPASShortwave.__new__(MPASShortwave)
    obj.cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        memGetInfo=lambda: (16 << 30, 32 << 30))),
        get_default_memory_pool=lambda: SimpleNamespace(free_bytes=lambda: 0))
    obj.batch_column_chunk = lambda nl: 256
    assert obj.sunlit_column_chunk(10000, 56) == 4096


@pytest.mark.parametrize('mode', ['single', 'shared-card'])
def test_spawned_suite_member_sets_policy_before_forecast(mode, monkeypatch):
    import os
    from woof.hex import multi_member
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'shared-card' if mode == 'single' else 'single')
    seen = []
    monkeypatch.setattr(multi_member, '_suite_member_main',
                        lambda member, log: seen.append(os.environ['WOOF_HEX_MEMBER_MEMORY']) or 0)
    with pytest.raises(SystemExit) as exc:
        multi_member._suite_member_process(object(), 'unused', mode)
    assert exc.value.code == 0 and seen == [mode]


@pytest.mark.parametrize('count,expected', [(1, 'single'), (2, 'shared-card')])
def test_roster_selects_memory_policy_without_changing_parent(count, expected, monkeypatch, tmp_path):
    import multiprocessing, os
    from woof.hex import multi_member, card_handoff
    members=[(f'm{i}', SimpleNamespace(out=tmp_path/f'm{i}')) for i in range(count)]
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'single')
    monkeypatch.setattr(card_handoff.CardGate, 'from_environ', lambda: None)
    monkeypatch.setattr(multi_member, 'roster_members', lambda args, door: members)
    monkeypatch.setattr(multi_member, 'admit_suite_group', lambda members, gate: {})
    seen=[]
    class Process:
        exitcode=0
        def __init__(self, *, target, args, name):
            assert target is multi_member._suite_member_process
            seen.append(args[2])
        def start(self): pass
        def join(self): pass
    monkeypatch.setattr(multiprocessing, 'get_context', lambda kind: SimpleNamespace(Process=Process))
    assert multi_member.run_suite_members(object(), object(), tmp_path, tmp_path) == 0
    assert seen == [expected]*count
    assert os.environ['WOOF_HEX_MEMBER_MEMORY'] == 'single'
