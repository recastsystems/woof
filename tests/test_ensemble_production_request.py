from collections import Counter
from contextvars import ContextVar
import threading

import pytest

from woof.ensemble.request import EnsembleRequest
from woof.ensemble.execution import execute_member_packing
from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent
from woof.ensemble.packing import CardBudget, pack_members


def test_request_has_no_member_cap_and_retains_source_and_threshold_contract():
    request = EnsembleRequest.from_mapping({"members": 513,
        "keep_member_files": True, "thresholds": {"wind10": [12, 22]},
        "member_device_ids": [0, 1]})
    assert request.members == 513
    assert request.receipt()["thresholds"] == {"wind10": [12.0, 22.0]}
    assert request.keep_member_files
    assert request.member_device_ids == (0, 1)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, {},
    {"members": 2, "sources": ["one"]}, {"members": 2, "member_device_ids": [0, 0]},
    {"members": 2, "keep_member_files": "yes"}, {"members": 2, "thresholds": {"gust": [float("nan")]}},
    {"members": 2, "unknown": 1}, {"members": 2, "base_seed": -1}])
def test_invalid_request_is_not_silently_changed(value):
    with pytest.raises((ValueError, TypeError)):
        EnsembleRequest.from_mapping(value)


def test_two_cards_run_one_job_each_and_sequential_waves_keep_all_members():
    model = EnsembleMemoryModel((MemoryComponent("state", "state", per_member_bytes=100),))
    plan = pack_members(9, (CardBudget(0, 200), CardBudget(1, 300)), model, batched=True)
    active = Counter()
    lock = threading.Lock()
    first_wave = threading.Barrier(2)
    scope = ContextVar("proof_scope", default=None)
    token = scope.set("original-context")
    def run(batch):
        assert scope.get() == "original-context"
        with lock:
            active[batch.device_id] += 1
            assert active[batch.device_id] == 1
        if batch.wave == 0:
            first_wave.wait(timeout=3)
        with lock:
            active[batch.device_id] -= 1
        return batch.member_indices
    try:
        results = execute_member_packing(plan, run)
    finally:
        scope.reset(token)
    assert sorted(member for batch, members in results for member in members) == list(range(9))
    assert plan.waves == 2
    assert not any(active.values())


def test_failure_finishes_other_owned_jobs_and_does_not_start_next_wave():
    model = EnsembleMemoryModel((MemoryComponent("state", "state", per_member_bytes=100),))
    plan = pack_members(5, (CardBudget(0, 100), CardBudget(1, 100)), model, batched=False)
    completed = []
    def run(batch):
        if batch.device_id == 0:
            raise RuntimeError("scientific failure")
        completed.append(batch.member_indices)
    with pytest.raises(RuntimeError, match="scientific failure"):
        execute_member_packing(plan, run)
    assert completed == [(1,)]


def test_memory_profile_reads_requested_card_without_repricing_default_callers(monkeypatch):
    from types import SimpleNamespace
    from woof.core import preflight
    seen = []
    def properties(device):
        seen.append(device)
        return dict(name=f"card-{device}", multiProcessorCount=120 + device,
                    maxThreadsPerMultiProcessor=2048)
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=False, 
        getDeviceProperties=properties, deviceGetLimit=lambda _: 1024)))
    monkeypatch.setattr(preflight, "read_compile_platform", lambda: ("120", "test"))
    default = preflight.local_memory_profile_from_device(cp)
    selected = preflight.local_memory_profile_from_device(cp, device_id=1)
    assert seen == [0, 1]
    assert default.name == "card-0"
    assert selected.name == "card-1"
    assert selected.multiprocessor_count == 121


def test_fast_card_refills_while_another_card_is_still_running():
    """A slow first batch must not idle a free card behind a global wave."""
    model = EnsembleMemoryModel((MemoryComponent("state", "state", per_member_bytes=100),))
    plan = pack_members(4, (CardBudget(0, 100), CardBudget(1, 100)), model, batched=False)
    fast_second = threading.Event()
    def run(batch):
        if batch.device_id == 1 and batch.wave == 0:
            assert fast_second.wait(3), "free card never received its queued member"
        if batch.device_id == 0 and batch.wave == 1:
            fast_second.set()
        return batch.member_indices
    result = execute_member_packing(plan, run)
    assert [members for batch, members in result] == [(0,), (1,), (2,), (3,)]
