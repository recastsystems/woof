"""The kernel prewarm replays exactly the compile calls a forecast made."""
from __future__ import annotations

import json
import time

import pytest

from woof.hex import kernel_prewarm, prep_cache


class _StubCompiler:
    def __init__(self, fail_on: str | None = None):
        self.calls = []
        self.fail_on = fail_on

    def _compile_with_cache_cuda(self, source, options, arch, extra_source=None, backend="nvrtc",
                                 enable_cooperative_groups=False, name_expressions=None,
                                 log_stream=None, cache_in_memory=False, jitify=False, to_ltoir=False):
        if source == self.fail_on:
            raise RuntimeError("nvrtc refused")
        self.calls.append((source, options, arch, extra_source, backend, enable_cooperative_groups,
                           None if name_expressions is None else tuple(name_expressions),
                           log_stream, cache_in_memory, jitify, to_ltoir))
        return object()

    @staticmethod
    def _get_nvrtc_version():
        return (13, 3)

    @staticmethod
    def _get_cupy_cache_key():
        return "stub-key"


@pytest.fixture
def stub(tmp_path, monkeypatch):
    monkeypatch.setenv(prep_cache.ENVIRONMENT, str(tmp_path / "prep"))
    monkeypatch.setenv("WOOF_HEX_CACHE_ROOT", str(tmp_path / "kernels"))
    monkeypatch.delenv("WOOF_HEX_KERNEL_CACHE_ROOT", raising=False)
    monkeypatch.delenv(kernel_prewarm.ENVIRONMENT, raising=False)
    # The manifest replay on its own; woof.hex.kernel_plan has its own tests.
    monkeypatch.setenv("WOOF_KERNEL_PLAN", "off")
    compiler = _StubCompiler()
    monkeypatch.setattr(kernel_prewarm, "_compiler", lambda: compiler)
    monkeypatch.setattr(kernel_prewarm, "_cupy_version", lambda: "14.2.0")
    monkeypatch.setattr(kernel_prewarm, "_use_device", lambda device: None)
    monkeypatch.setattr(kernel_prewarm, "_context_token", lambda: (0, 101), raising=False)
    monkeypatch.setattr(kernel_prewarm.atexit, "register", lambda function: None)
    monkeypatch.setattr(kernel_prewarm, "_recorded", {})
    monkeypatch.setattr(kernel_prewarm, "_state", {"recording": False, "replay": None})
    monkeypatch.setattr(kernel_prewarm, "_inflight", {})
    return compiler


def _wait(record):
    for _ in range(200):
        if record["seconds"] is not None:
            return
        time.sleep(0.01)
    raise AssertionError("replay did not finish")


def test_recorded_calls_replay_with_identical_arguments(stub):
    kernel_prewarm.install_recorder()
    recorder = stub._compile_with_cache_cuda
    recorder("extern \"C\" __global__ void a(){}", ("--std=c++17", "--fmad=false"), None)
    recorder("extern \"C\" __global__ void b(){}", ("--fmad=false",), "120", None, "nvrtc", False,
             ["b<float>"], None, False, False, False)
    recorder("in memory only", (), None, cache_in_memory=True)
    first = list(stub.calls)
    kernel_prewarm.save_manifest()
    document = json.loads(kernel_prewarm.manifest_path().read_text())
    assert document["schema"] == "gpuwm.kernel-pack/v1"
    rows = [row["arguments"] for row in document["entries"]]
    assert len(rows) == 2

    stub.calls.clear()
    kernel_prewarm._state["replay"] = None
    record = kernel_prewarm.start_replay(device_id=0)
    _wait(record)
    assert record["entries"] == 2 and record["compiled_or_found"] == 2 and record["failed"] == 0
    assert sorted(stub.calls) == sorted(first[:2])


def test_failed_replay_is_counted_not_raised(stub):
    kernel_prewarm.install_recorder()
    stub._compile_with_cache_cuda("good", ("-a",), None)
    stub._compile_with_cache_cuda("bad", ("-a",), None)
    kernel_prewarm.save_manifest()
    stub.fail_on = "bad"
    kernel_prewarm._state["replay"] = None
    record = kernel_prewarm.start_replay(device_id=0)
    _wait(record)
    assert record["compiled_or_found"] == 1 and record["failed"] == 1


def test_off_switch_neither_records_nor_replays(stub, monkeypatch):
    monkeypatch.setenv(kernel_prewarm.ENVIRONMENT, "off")
    original = stub._compile_with_cache_cuda
    kernel_prewarm.install_recorder()
    assert stub._compile_with_cache_cuda == original
    assert kernel_prewarm.start_replay(device_id=0) is None


def test_damaged_manifest_reads_as_empty(stub):
    path = kernel_prewarm.manifest_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json\n")
    record = kernel_prewarm.start_replay(device_id=0)
    assert record["entries"] == 0


def test_legacy_image_manifest_remains_replayable(stub):
    kernel_prewarm.install_recorder()
    stub._compile_with_cache_cuda("legacy", ("--fmad=false",), "120")
    entry = next(iter(kernel_prewarm._recorded.values()))
    path = kernel_prewarm.manifest_path().with_suffix(".jsonl")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(entry) + "\n")
    assert kernel_prewarm._load_manifest(kernel_prewarm.manifest_path()) == [entry]


def test_shared_pack_codec_and_hex_reader_agree(stub):
    kernel_pack = pytest.importorskip("woof.hex.kernel_pack")
    kernel_prewarm.install_recorder()
    stub._compile_with_cache_cuda("source", ("--fmad=false",), "120")
    kernel_prewarm.save_manifest()
    path = kernel_prewarm.manifest_path()
    entries = kernel_pack.read(path, kernel_prewarm._compiler_identity())
    assert entries == kernel_prewarm._load_manifest(path)
    kernel_pack.write(path, kernel_prewarm._compiler_identity(), entries)
    assert kernel_prewarm._load_manifest(path) == entries


def test_forecast_takes_back_a_module_the_pool_has_not_started(stub, monkeypatch):
    import threading

    kernel_prewarm.install_recorder()
    recorder = stub._compile_with_cache_cuda
    for name in ("first", "second", "third"):
        recorder(name, ("-a",), None)
    kernel_prewarm.save_manifest()
    gate = threading.Event()
    original = kernel_prewarm._state["original"]

    def slow(source, *rest):
        if source == "first":
            gate.wait(5)
        return original(source, *rest)

    kernel_prewarm._state["original"] = slow
    monkeypatch.setenv("WOOF_KERNEL_PLAN_WORKERS", "1")  # one worker
    stub.calls.clear()
    kernel_prewarm._state["replay"] = None
    record = kernel_prewarm.start_replay(device_id=0)
    recorder("third", ("-a",), None)  # the forecast reaches it before the pool does
    gate.set()
    _wait(record)
    assert record.get("taken_back") == 1
    assert [call[0] for call in stub.calls].count("third") == 1


def test_background_setup_overlaps_foreground_and_preserves_device(stub, monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()
    calls, devices = [], []
    original_thread = threading.get_ident()
    monkeypatch.setattr(kernel_prewarm.kernel_plan, "_current_device", lambda: 7)
    monkeypatch.setattr(kernel_prewarm, "_use_device", lambda device: devices.append((device, threading.get_ident())))

    def held(device):
        calls.append(device)
        entered.set()
        assert release.wait(5)
        return {"entries": 17}

    monkeypatch.setattr(kernel_prewarm, "_start_replay", held)
    kernel_prewarm.install_recorder()
    future = kernel_prewarm.start_replay_async()
    try:
        assert entered.wait(5)
        assert not future.done()
        assert kernel_prewarm.start_replay_async(9) is future
        stub._compile_with_cache_cuda("foreground", (), "120")
        assert len(stub.calls) == 1
        assert kernel_prewarm.receipt()["setup"]["status"] == "running"
    finally:
        release.set()
    assert future.result(5) == {"entries": 17}
    assert kernel_prewarm.start_replay() == {"entries": 17}
    assert calls == [7]
    assert devices[0][0] == 7 and devices[0][1] != original_thread
    assert kernel_prewarm.receipt()["setup"]["status"] == "complete"


def test_background_setup_failure_is_loud_and_foreground_survives(stub, monkeypatch):
    def broken(device):
        raise ValueError("damaged plan fixture")
    monkeypatch.setattr(kernel_prewarm, "_start_replay", broken)
    kernel_prewarm.install_recorder()
    with pytest.warns(RuntimeWarning, match="foreground fallback remains active"):
        assert kernel_prewarm.start_replay_async(0).result(5) is None
    stub._compile_with_cache_cuda("foreground after error", (), "120")
    assert len(stub.calls) == 1
    record = kernel_prewarm.receipt()["setup"]
    assert record["status"] == "failed"
    assert record["error"] == "ValueError: damaged plan fixture"
    assert record["seconds"] >= 0


def test_background_disabled_does_not_capture_device_or_start_worker(stub, monkeypatch):
    monkeypatch.setenv(kernel_prewarm.ENVIRONMENT, "off")
    def forbidden():
        pytest.fail("disabled prewarm queried CUDA")
    monkeypatch.setattr(kernel_prewarm.kernel_plan, "_current_device", forbidden)
    assert kernel_prewarm.start_replay_async() is None
    assert "setup_future" not in kernel_prewarm._state


def test_synchronous_replay_joins_active_background_setup(stub, monkeypatch):
    import concurrent.futures
    import threading
    entered, release, joining = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def held(device):
        calls.append(device)
        entered.set()
        assert release.wait(5)
        return {"entries": 11}
    monkeypatch.setattr(kernel_prewarm, "_start_replay", held)
    pending = kernel_prewarm.start_replay_async(0)
    assert entered.wait(5)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        def join():
            joining.set()
            return kernel_prewarm.start_replay()
        result = pool.submit(join)
        try:
            assert joining.wait(5)
            assert not result.done()
        finally:
            release.set()
        assert result.result(5) == pending.result(5) == {"entries": 11}
    assert calls == [0]


def test_completed_replay_hands_module_to_forecast_without_second_load(stub):
    kernel_prewarm.install_recorder()
    recorder = stub._compile_with_cache_cuda
    recorder("module", ("--std=c++17",), "120", name_expressions=["f<float>"])
    kernel_prewarm.save_manifest()
    stub.calls.clear()
    record = kernel_prewarm.start_replay(device_id=0)
    _wait(record)
    future = next(iter(kernel_prewarm._inflight.values()))
    module = recorder("module", ("--std=c++17",), "120", name_expressions=["f<float>"])
    assert len(stub.calls) == 1
    assert module is future.result()[0]
    assert record["served_replay"] == 1
    assert not kernel_prewarm._inflight


def test_replay_module_never_crosses_cuda_context(stub, monkeypatch):
    kernel_prewarm.install_recorder()
    recorder = stub._compile_with_cache_cuda
    recorder("module", (), "120")
    kernel_prewarm.save_manifest()
    stub.calls.clear()
    record = kernel_prewarm.start_replay(device_id=0)
    _wait(record)
    for token in ((1, 101), (0, 202), None):
        # Recreate the finished slot to exercise each distinct refusal boundary.
        future = next(iter(kernel_prewarm._inflight.values())) if token == (1, 101) else saved
        saved = future
        digest = kernel_prewarm._digest(next(iter(kernel_prewarm._recorded.values())))
        kernel_prewarm._inflight[digest] = future
        monkeypatch.setattr(kernel_prewarm, "_context_token", lambda token=token: token)
        recorder("module", (), "120")
    assert len(stub.calls) == 4
    assert record.get("served_replay", 0) == 0
    assert record["replay_context_mismatch"] == 3


def test_active_replay_handoff_waits_once(stub, monkeypatch):
    import threading
    import concurrent.futures
    entered, release = threading.Event(), threading.Event()
    kernel_prewarm.install_recorder()
    recorder = stub._compile_with_cache_cuda
    recorder("module", (), "120")
    kernel_prewarm.save_manifest()
    original = kernel_prewarm._state["original"]
    def held(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    kernel_prewarm._state["original"] = held
    stub.calls.clear()
    record = kernel_prewarm.start_replay(device_id=0)
    assert entered.wait(5)
    with concurrent.futures.ThreadPoolExecutor(1) as callers:
        result = callers.submit(recorder, "module", (), "120")
        try:
            # The running compile cannot be cancelled or replaced.
            assert not result.done()
        finally:
            release.set()
        assert result.result(5) is not None
    _wait(record)
    assert len(stub.calls) == 1
    assert record["waited_for"] == record["served_replay"] == 1


def test_failed_or_changed_request_replay_retains_original_call(stub):
    kernel_prewarm.install_recorder()
    recorder = stub._compile_with_cache_cuda
    recorder("module", ("-a",), "120")
    kernel_prewarm.save_manifest()
    stub.calls.clear()
    stub.fail_on = "module"
    record = kernel_prewarm.start_replay(device_id=0)
    _wait(record)
    assert record["failed"] == 1
    stub.fail_on = None
    recorder("module", ("-a",), "120")
    recorder("module", ("-b",), "120")
    assert [call[1] for call in stub.calls] == [("-a",), ("-b",)]
    assert record.get("served_replay", 0) == 0


def test_host_cache_off_keeps_kernel_replay_isolated(stub, monkeypatch, tmp_path):
    from woof.hex import kernel_warm
    monkeypatch.setenv(prep_cache.ENVIRONMENT, 'off')
    monkeypatch.setenv('WOOF_HEX_CACHE_ROOT', str(tmp_path / 'kernels'))
    assert not prep_cache.enabled()
    assert kernel_prewarm.enabled()
    assert kernel_prewarm._plan_directory() == tmp_path / 'kernels/kernel-plans'
    assert kernel_warm._root() == tmp_path / 'kernels/kernel-packs'
    kernel_prewarm.install_recorder()
    stub._compile_with_cache_cuda('source', ('--fmad=false',), '120')
    kernel_prewarm.save_manifest()
    assert kernel_prewarm.manifest_path().is_relative_to(tmp_path)
    record = kernel_prewarm.start_replay(device_id=0)
    _wait(record)
    assert record['compiled_or_found'] == 1
    monkeypatch.setenv(kernel_prewarm.ENVIRONMENT, 'off')
    assert not kernel_prewarm.enabled()
    assert kernel_warm._root() is None
