"""The kernel plan compiles a run's recorded kernel set ahead, on a pool, with identical arguments.

Breakage these tests prevent: a cold-cache forecast holding ONE core for minutes
before step 1 while its card idles (x1.40962 on an RTX 4090: 111.5 s of step 1
at 1.1 cores, 2026-10-16), and any change to what NVRTC is asked to compile.
"""
from __future__ import annotations

import json
import concurrent.futures
import threading
import time
import sys
from types import SimpleNamespace

import pytest

from woof import kernel_plan


class _StubCompiler:
    """The NVRTC funnel's signature; returns a deterministic fake image."""

    def __init__(self, arch="89"):
        self.calls = []
        self.arch = arch
        self.gate = None
        self.fail_on = None
        self._lock = threading.Lock()

    def _compile_using_nvrtc_no_warning(self, source, options=(), arch=None, filename="kern.cu",
                                        name_expressions=None, log_stream=None, cache_in_memory=False,
                                        jitify=None, method=None):
        if self.gate is not None and source == "slow":
            self.gate.wait(5)
        if source == self.fail_on:
            raise RuntimeError("nvrtc refused")
        resolved = self.arch if arch is None else str(arch)
        with self._lock:
            self.calls.append((source, tuple(options), resolved, filename, cache_in_memory,
                               threading.current_thread().name))
        return (f"image:{source}:{options}:{resolved}:{filename}:{cache_in_memory}".encode(),
                {"k": "mangled"} if name_expressions else None)

    def _get_arch(self):
        return self.arch

    @staticmethod
    def _get_nvrtc_version():
        return (13, 3)

    @staticmethod
    def _get_cupy_cache_key():
        return "stub-key"


def _fresh(monkeypatch, compiler):
    monkeypatch.setattr(kernel_plan, "_compiler", lambda: compiler)
    monkeypatch.setattr(kernel_plan, "compiler_identity", lambda: {"stub": 1})
    monkeypatch.setattr(kernel_plan, "_current_device", lambda: None)
    monkeypatch.setattr(kernel_plan.atexit, "register", lambda function: None)
    monkeypatch.setattr(kernel_plan, "_flight", {})
    monkeypatch.setattr(kernel_plan, "_requested", {})
    monkeypatch.setattr(kernel_plan, "_state", {"installed": False, "plan": None, "tag": None, "tag_text": None,
                                                "device_arch": None, "replay": None, "counts": {}})


@pytest.fixture
def plan_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(kernel_plan.ENVIRONMENT, raising=False)
    monkeypatch.delenv(kernel_plan.WORKERS_ENVIRONMENT, raising=False)
    return tmp_path / "plans"


def _record_run(monkeypatch, plan_dir, compiler, requests, tag="forecast --mesh"):
    _fresh(monkeypatch, compiler)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text=tag)
    for args, kwargs in requests:
        compiler._compile_using_nvrtc_no_warning(*args, **kwargs)
    kernel_plan.save()


def _wait(predicate):
    for _ in range(500):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("pool did not finish")


def test_unrecorded_dynamic_requests_share_the_process_compiler(monkeypatch, plan_dir):
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text="new-shape")
    kernel_plan._state.update(compiler_pool=object(), replay={})
    submitted = []
    def compile_request(arguments, arch, record):
        submitted.append((arguments, arch))
        return b"compiled-in-worker", None
    monkeypatch.setattr(kernel_plan, "_submit_and_wait", compile_request)
    result = compiler._compile_using_nvrtc_no_warning("new generated reduction", arch="89")
    assert result == (b"compiled-in-worker", None)
    assert not compiler.calls
    assert submitted[0][0]["source"] == "new generated reduction"
    assert submitted[0][1] == "89"
    assert kernel_plan._state["counts"]["dynamic_pool_compiles"] == 1


def test_startup_keeps_the_process_compiler_for_first_step_discoveries(monkeypatch, plan_dir):
    from woof import kernel_prewarm, local_gpu
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text='first-step')
    record = {}
    kernel_plan._state.update(compiler_pool=SimpleNamespace(shutdown=lambda **kw: None), replay=record)
    monkeypatch.setattr(local_gpu, 'no_local_gpu', lambda: False)
    monkeypatch.setattr(kernel_prewarm, 'plan_tag', lambda experiment: 'first-step')
    monkeypatch.setattr(kernel_prewarm, 'submit_static', lambda *args, **kwargs: 0)
    monkeypatch.setattr(kernel_plan, 'start', lambda: record)
    monkeypatch.setattr(kernel_plan, '_submit_and_wait', lambda *args: (b'worker-image', None))
    try:
        assert kernel_prewarm.start_plan(SimpleNamespace(domains=())) is record
        assert compiler._compile_using_nvrtc_no_warning('first-step generated reduction') == (b'worker-image', None)
        assert not compiler.calls
    finally:
        kernel_plan.close(wait=True)


def test_concurrent_cold_install_does_not_wrap_its_own_funnel(monkeypatch, plan_dir):
    """Two members used to nest funnels and wait forever inside cuda_cache.

    Pause the second installer after its initial check, until the first has
    published its funnel. The CuPy bridge then rereads that funnel as its
    original compiler. A serialized install never reaches the second identity
    call. Check the invariant before launching work so a regression fails
    without leaving a deadlocked thread in the test runner.
    """
    from woof import kernel_aot, kernel_warm_pack
    from woof.core.device_cache import cuda_cache

    here = _StubCompiler()
    here.__name__ = "cupy.cuda.compiler"
    original = here._compile_using_nvrtc_no_warning
    _fresh(monkeypatch, here)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda: SimpleNamespace(id=0),
        get_current_stream=lambda: SimpleNamespace(ptr=1))))
    monkeypatch.setattr(kernel_aot, "install", lambda cp: None)
    monkeypatch.setattr(kernel_plan, "_warm_images", lambda: None)
    first_entered, second_entered, first_done = (threading.Event() for _ in range(3))
    errors, installed = [], []

    def identity():
        if threading.current_thread().name == "member-1":
            second_entered.set()
            assert first_done.wait(3)
        return {"stub": 1}

    def metadata(*args):
        if threading.current_thread().name == "member-0":
            first_entered.set()
            # With the fix member 1 waits on the installation lock instead.
            second_entered.wait(0.2)

    monkeypatch.setattr(kernel_plan, "compiler_identity", identity)
    monkeypatch.setattr(kernel_warm_pack, "install_metadata", metadata)

    def install(index):
        try:
            installed.append(kernel_plan.install(plan_dir=plan_dir, tag_text="cold members"))
        except BaseException as error:
            errors.append(error)
        finally:
            if index == 0:
                first_done.set()

    threads = [threading.Thread(target=install, args=(i,), name=f"member-{i}", daemon=True)
               for i in range(2)]
    threads[0].start()
    assert first_entered.wait(3)
    threads[1].start()
    for thread in threads:
        thread.join(3)
        assert not thread.is_alive(), "concurrent installation deadlocked"
    assert not errors and installed == [True, True]
    assert kernel_plan._state["original"] == original, "funnel would wait on its own future"
    assert here._compile_using_nvrtc_no_warning.__wrapped__ == original
    assert not kernel_plan._flight and not plan_dir.exists()

    @cuda_cache()
    def load():
        return here._compile_using_nvrtc_no_warning("cold", ("--std=c++17",), None,
                                                   "cold.cu", ["k"])

    # Static capture and foreground first use share the cold plan and the
    # actual device-cache factory lock, returning the exact compiler bytes.
    barrier = threading.Barrier(2)
    results = []

    def member(warm):
        try:
            barrier.wait(3)
            if warm:
                kernel_plan.ahead(load)
            results.append(load())
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=member, args=(warm,), daemon=True) for warm in (True, False)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
        assert not thread.is_alive(), "cold physics factory deadlocked"
    assert not errors
    expected = _StubCompiler()._compile_using_nvrtc_no_warning(
        "cold", ("--std=c++17",), None, "cold.cu", ["k"])
    assert results == [expected, expected]
    assert len(here.calls) == 1


def test_next_run_compiles_the_plan_ahead_and_serves_identical_results(monkeypatch, plan_dir):
    requests = [(("A", ("--std=c++17", "--fmad=false"), None, "a.cu"), {}),
                (("B", ("--std=c++17",), "89", "b.cu", ["f<float>"]), {}),
                (("C", ("--std=c++17",)), {"cache_in_memory": True})]
    first = _StubCompiler()
    _record_run(monkeypatch, plan_dir, first, requests)
    serial = [(call[0], call[1], call[2]) for call in first.calls]
    document = json.loads(next(plan_dir.rglob("*.json")).read_text())
    assert len(document["entries"]) == 3 and all(row["arguments"]["arch"] == "device" for row in document["entries"])

    second = _StubCompiler()
    _fresh(monkeypatch, second)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["entries"] == 3 and record["plan_source"] == "tag"
    _wait(lambda: record["pool_compiled"] == 3)
    assert sorted((c[0], c[1], c[2]) for c in second.calls) == sorted(serial)
    assert all(c[5].startswith(kernel_plan.POOL_PREFIX) for c in second.calls)
    results = [second._compile_using_nvrtc_no_warning(*args, **kwargs) for args, kwargs in requests]
    assert len(second.calls) == 3  # served from the pool, not compiled again
    expected = [first.__class__()._compile_using_nvrtc_no_warning(*args, **kwargs) for args, kwargs in requests]
    assert results == expected
    assert kernel_plan.receipt()["calls"]["served"] == 3


def test_a_plan_recorded_on_another_architecture_compiles_for_this_card(monkeypatch, plan_dir):
    old, current = _cupy_program("A-120"), _cupy_program("A-89")
    _record_run(monkeypatch, plan_dir, _StubCompiler(arch="120"), [(("A", ("-x",), "120", old), {})])
    monkeypatch.setattr(kernel_plan, "_cupy_entry_elsewhere", lambda arguments, entries, arch: current[:-3])
    here = _StubCompiler(arch="89")
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    assert here.calls[0][2] == "89"
    # CuPy's architecture-bound hash is part of the image too. Replay must use this card's name.
    assert here.calls[0][3] == current
    got = here._compile_using_nvrtc_no_warning("A", ("-x",), "89", current)
    assert got == _StubCompiler()._compile_using_nvrtc_no_warning("A", ("-x",), "89", current)
    assert len(here.calls) == 1


def test_program_name_is_matched_without_asserts_or_debug_options(monkeypatch, plan_dir):
    source = 'static __device__ float table[] = {1.0f};'
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [((source, ("-x",), None, "one.cu"), {})])
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    here._compile_using_nvrtc_no_warning(source, ("-x",), None, "two.cu")
    assert len(here.calls) == 2  # NVRTC embeds the program name in internal symbols too.
    here._compile_using_nvrtc_no_warning(source, ("-x",), None, "one.cu")
    assert len(here.calls) == 2


def test_the_run_takes_back_a_compile_the_pool_has_not_started(monkeypatch, plan_dir):
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("slow", ()), {}), (("queued", ()), {})])
    here = _StubCompiler()
    here.gate = threading.Event()
    _fresh(monkeypatch, here)
    monkeypatch.setenv(kernel_plan.WORKERS_ENVIRONMENT, "1")
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    kernel_plan._state["device_arch"] = None
    record = kernel_plan.start()
    kernel_plan.close()
    here._compile_using_nvrtc_no_warning("queued", ())  # reached before the one worker is free
    here.gate.set()
    _wait(lambda: record["pool_compiled"] == 1)
    assert [c[0] for c in here.calls].count("queued") == 1
    assert kernel_plan.receipt()["calls"]["taken_back"] == 1


def test_a_failed_pool_compile_is_compiled_again_by_the_run(monkeypatch, plan_dir):
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("bad", ()), {})])
    here = _StubCompiler()
    here.fail_on = "bad"
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_failed"] == 1)
    with pytest.raises(RuntimeError, match="nvrtc refused"):
        here._compile_using_nvrtc_no_warning("bad", ())


def test_unknown_shape_does_not_scan_other_plans(monkeypatch, plan_dir):
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("A", ()), {})], tag="one")
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("B", ()), {})], tag="two")
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="three")
    from pathlib import Path
    monkeypatch.setattr(Path, "glob", lambda *a: pytest.fail("plan directory scan"))
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["plan_source"] == "none" and record["entries"] == 0
    assert not here.calls


def test_off_switch_leaves_the_compiler_untouched(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    monkeypatch.setenv(kernel_plan.ENVIRONMENT, "off")
    original = here._compile_using_nvrtc_no_warning
    assert not kernel_plan.install(plan_dir=plan_dir)
    assert here._compile_using_nvrtc_no_warning == original
    assert kernel_plan.start() is None


def test_damaged_plan_reads_as_empty(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="x")
    kernel_plan._state["plan"].parent.mkdir(parents=True)
    kernel_plan._state["plan"].write_text("{not json\n")
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["entries"] == 0


def test_tag_ignores_paths_and_numbers():
    a = kernel_plan.default_tag_text(["forecast", "--init", "/a/x1.40962.init.nc", "--hours", "24", "--no-products"])
    b = kernel_plan.default_tag_text(["forecast", "--init", "/b/other.nc", "--hours", "3", "--no-products"])
    assert a == b and "--no-products" in a


def _cupy_program(tag):
    """CuPy's cached route names its program after the disk entry: <sha1>.cubin.cu."""
    import hashlib
    return hashlib.sha1(tag.encode()).hexdigest() + ".cubin.cu"


def test_a_row_its_route_finds_on_disk_is_not_compiled_ahead(monkeypatch, plan_dir, tmp_path):
    # Breakage: with a warm cache and plans present, warm-kernels took 122.5 s and 78 NVRTC
    # compiles against 2.5 s and 0 without the plans (RTX 4090, 2026-10-17).
    cache = tmp_path / "cupy-cache"
    cache.mkdir()
    recorder = _StubCompiler()
    recorder.get_cache_dir = lambda: str(cache)
    program = _cupy_program("A-89")
    _record_run(monkeypatch, plan_dir, recorder, [(("A", ("-x",), None, program), {}),
                                                  (("M", ("-x",)), {"cache_in_memory": True})])
    rows = json.loads(next(plan_dir.rglob("*.json")).read_text())["entries"]
    entries = {row["arguments"]["source"]: row.get("cache_entries") for row in rows}
    assert entries == {"A": {"89": program[:-3]}, "M": None}
    (cache / program[:-3]).write_bytes(b"cubin")  # what CuPy stored after the compile

    here = _StubCompiler()
    here.get_cache_dir = lambda: str(cache)
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    assert record["pool_skipped_cached"] == 1
    assert [call[0] for call in here.calls] == ["M"]  # the in-memory route still compiles ahead

    (cache / program[:-3]).unlink()  # a cold cache: the plan compiles the set again
    cold = _StubCompiler()
    cold.get_cache_dir = lambda: str(cache)
    _fresh(monkeypatch, cold)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 2)
    assert record["pool_skipped_cached"] == 0


def probe_for_test(arguments, arch):
    return f"probe/{arguments['source']}-{arch}.image"


def test_a_named_entry_and_its_probe_cover_another_architecture(monkeypatch, plan_dir, tmp_path):
    cache = tmp_path / "cupy-cache"
    (cache / "probe").mkdir(parents=True)
    recorder = _StubCompiler(arch="89")
    recorder.get_cache_dir = lambda: str(cache)
    _fresh(monkeypatch, recorder)
    kernel_plan.install(plan_dir=plan_dir, tag_text="t")
    with kernel_plan.cache_entry(f"{__name__}:probe_for_test", cache / "probe" / "D-89.image"):
        recorder._compile_using_nvrtc_no_warning("D", ("-x",), None, "d.cu")
    kernel_plan.save()
    row = json.loads(next(plan_dir.rglob("*.json")).read_text())["entries"][0]
    assert row["cache_entries"] == {"89": "probe/D-89.image"}  # stored relative to the cache root
    (cache / "probe" / "D-120.image").write_bytes(b"x")  # baked for sm_120, never recorded there
    here = _StubCompiler(arch="120")
    here.get_cache_dir = lambda: str(cache)
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="t")
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["pool_skipped_cached"] == 1 and here.calls == []


def test_rebind_gives_each_configuration_its_own_plan(monkeypatch, plan_dir):
    # Breakage: a warm-kernels worker wrote every profile it warmed under its first profile's tag.
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="one")
    here._compile_using_nvrtc_no_warning("A", ())
    assert kernel_plan.rebind("two")
    here._compile_using_nvrtc_no_warning("B", ())
    kernel_plan.save()
    plans = {json.loads(path.read_text())["tag_text"]: [row["arguments"]["source"] for row in
             json.loads(path.read_text())["entries"]] for path in plan_dir.rglob("*.json")}
    assert plans == {"one": ["A"], "two": ["B"]}


def _loader(compiler, source, records):
    """A loader's uncached body: its NVRTC request, then the records a real load writes."""

    def load(name):
        image = compiler._compile_using_nvrtc_no_warning(source + name, ("--std=c++17", "--ftz=false"), None,
                                                         f"{name}.cu")
        kernel_plan.refuse_while_capturing()
        records.append(name)
        return image
    return load


def test_a_first_run_compiles_its_static_plan_ahead_and_serves_identical_results(monkeypatch, plan_dir):
    """No recorded plan (a fresh box): the static plan's captures compile on the pool, the run is served."""
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text="first run")
    record = kernel_plan.start()
    assert record["entries"] == 0 and record["plan_source"] == "none"
    records = []
    load = _loader(here, "src:", records)
    futures = [kernel_plan.submit(kernel_plan.ahead, load, name) for name in ("noah", "smag2d", "ysu_480")]
    kernel_plan.close()
    assert sorted(f.result(5) for f in futures) == ["compiled"] * 3
    assert all(c[5].startswith(kernel_plan.POOL_PREFIX) for c in here.calls) and len(here.calls) == 3
    assert records == []  # the capture wrote nothing a real load writes
    results = [load(name) for name in ("noah", "smag2d", "ysu_480")]
    assert len(here.calls) == 3  # served from the table, not compiled again
    assert results == [_StubCompiler()._compile_using_nvrtc_no_warning(
        "src:" + name, ("--std=c++17", "--ftz=false"), None, f"{name}.cu") for name in ("noah", "smag2d", "ysu_480")]
    assert records == ["noah", "smag2d", "ysu_480"]
    assert kernel_plan.receipt()["replay"]["static_compiled"] == 3


def test_a_static_capture_whose_image_is_on_disk_compiles_and_records_nothing(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="warm")
    kernel_plan.start()
    records = []

    def cached_load(name):  # the route read its image from disk: no NVRTC request
        kernel_plan.refuse_while_capturing()
        records.append(name)

    assert kernel_plan.submit(kernel_plan.ahead, cached_load, "noah").result(5) == "cached"
    kernel_plan.close()
    assert here.calls == [] and records == []
    cached_load("noah")
    assert records == ["noah"]  # outside a capture the guard is inert


def test_a_static_capture_the_run_requests_differently_is_compiled_by_the_run(monkeypatch, plan_dir):
    """A parameter set declared after the capture changes the bytes: the run misses and compiles inline."""
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="declared later")
    kernel_plan.start()
    records = []
    assert kernel_plan.submit(kernel_plan.ahead, _loader(here, "default:", records), "mynn").result(5) == "compiled"
    kernel_plan.close()
    _loader(here, "declared:", records)("mynn")
    assert [c[0] for c in here.calls] == ["default:mynn", "declared:mynn"]
    assert here.calls[1][5] == threading.current_thread().name


def test_a_static_capture_never_compiles_twice_what_the_run_already_compiles(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="raced")
    kernel_plan.start()
    records = []
    load = _loader(here, "src:", records)
    load("noah")  # the run got there first
    assert kernel_plan.submit(kernel_plan.ahead, load, "noah").result(5) == "known"
    kernel_plan.close()
    assert len(here.calls) == 1 and records == ["noah"]


def test_a_static_capture_that_fails_leaves_the_compile_to_the_run(monkeypatch, plan_dir):
    here = _StubCompiler()
    here.fail_on = "src:bad"
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="failing")
    kernel_plan.start()
    records = []
    load = _loader(here, "src:", records)
    assert kernel_plan.submit(kernel_plan.ahead, load, "bad").result(5) == "failed"
    kernel_plan.close()
    with pytest.raises(RuntimeError, match="nvrtc refused"):
        load("bad")
    assert records == []


@pytest.mark.parametrize("running", [False, True])
def test_process_future_cancels_only_an_actual_queued_compile(running):
    pending = concurrent.futures.Future()
    if running:
        pending.set_running_or_notify_cancel()
    record = dict(pool_failed=0, pool_compiled=0, pool_seconds=0.0,
                  pool_cpu_seconds=0.0, pool_processes=[])
    result = kernel_plan._ProcessCompileFuture(pending, record)
    pending.add_done_callback(result.completed)
    assert result.cancel() is (not running)
    if running:
        pending.set_result(((b"image", {"k": "mangled"}),
                            dict(pid=123, seconds=0.4, cpu_seconds=0.3)))
        assert result.result() == (b"image", {"k": "mangled"})
        assert record["pool_compiled"] == 1
        assert record["pool_cpu_seconds"] == 0.3
    else:
        assert result.cancelled() and record["pool_compiled"] == 0


def test_process_compile_error_reaches_the_requester():
    pending = concurrent.futures.Future()
    record = dict(pool_failed=0)
    result = kernel_plan._ProcessCompileFuture(pending, record)
    pending.add_done_callback(result.completed)
    pending.set_exception(RuntimeError("nvrtc refused"))
    with pytest.raises(RuntimeError, match="nvrtc refused"):
        result.result()
    assert record["pool_failed"] == 1


@pytest.mark.parametrize("trace", [False, True])
def test_recorded_process_jobs_leave_warm_threads_free(monkeypatch, plan_dir, trace):
    if trace:
        monkeypatch.setenv("WOOF_KERNEL_PLAN_TRACE", "1")
    else:
        monkeypatch.delenv("WOOF_KERNEL_PLAN_TRACE", raising=False)
    _record_run(monkeypatch, plan_dir, _StubCompiler(),
                [(("A", ()), {}), (("B", ()), {})])
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    monkeypatch.setenv(kernel_plan.WORKERS_ENVIRONMENT, "2")
    monkeypatch.setattr(kernel_plan, "_process_compiler_identity", lambda arch: {"worker": 1})
    jobs = []

    class ProcessPool:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, call, arguments, arch, identity, **kwargs):
            assert kwargs == ({"trace": True} if trace else {})
            pending = concurrent.futures.Future()
            pending.set_running_or_notify_cancel()
            jobs.append((pending, arguments))
            return pending

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", ProcessPool)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    try:
        # Both compilers are held running. A GPU warm callback must still
        # get a thread, and the flight table must own the process proxies.
        assert kernel_plan.submit(lambda: "warm").result(timeout=1) == "warm"
        assert len(jobs) == 2
        assert all(isinstance(f, kernel_plan._ProcessCompileFuture)
                   for f in kernel_plan._flight.values())
        assert all(not f.cancel() for f in kernel_plan._flight.values())
    finally:
        for pending, arguments in jobs:
            pending.set_result(((arguments["source"].encode(), {"k": "mangled"}),
                                dict(pid=123, seconds=0.2, cpu_seconds=0.1,
                                     **({"compile_begin_mono_ns": 100,
                                         "compile_end_mono_ns": 200} if trace else {}))))
        kernel_plan.close(wait=True)
    assert here._compile_using_nvrtc_no_warning("A", ()) == (b"A", {"k": "mangled"})
    assert here.calls == []
    assert record["pool_compiled"] == 2 and record["pool_failed"] == 0

    if trace:
        assert len(record["compile_intervals"]) == 2
        assert all(row["request_sha256"] for row in record["compile_intervals"])
    else:
        assert "compile_intervals" not in record


def test_pack_covered_replay_is_left_to_the_verified_loader(monkeypatch, plan_dir):
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("A", ()), {})])
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    monkeypatch.setattr(kernel_plan, "_aot_coverage", lambda: {"held"})
    monkeypatch.setattr(kernel_plan, "_aot_cached", lambda *args: True)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["pool_skipped_aot"] == 1
    assert record["pool_compiled"] == 0
    assert here.calls == []
    # A scheduling hint never supplies bytes: the real caller still compiles
    # normally if its high-level pack loader fails the complete contract.
    here._compile_using_nvrtc_no_warning("A", ())
    assert len(here.calls) == 1


def test_plan_tag_separates_effective_engine_numerics(monkeypatch):
    from woof import numerics
    monkeypatch.setattr(numerics, "_mode", "fast")
    fast = kernel_plan.default_tag_text(())
    monkeypatch.setattr(numerics, "_mode", "strict")
    strict = kernel_plan.default_tag_text(())
    assert fast != strict


def test_rebind_drains_closed_pool_before_replacing_compiler_state(monkeypatch, plan_dir):
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="one")
    kernel_plan.start()
    entered, release = threading.Event(), threading.Event()

    def capture():
        entered.set()
        assert release.wait(5)
        return kernel_plan.ahead(_loader(here, "old:", []), "noah")

    pending = kernel_plan.submit(capture)
    assert entered.wait(5)
    kernel_plan.close()
    rebound = threading.Event()
    thread = threading.Thread(target=lambda: (kernel_plan.rebind("two"), rebound.set()))
    thread.start()
    assert not rebound.wait(0.05)
    assert kernel_plan._state["tag_text"] == "one"
    release.set()
    thread.join(5)
    assert rebound.is_set() and pending.result() == "compiled"
    assert kernel_plan._state["tag_text"] == "two"


def test_async_close_and_wait_have_one_process_shutdown_owner(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    class ProcessPool:
        calls = 0
        def shutdown(self, wait):
            self.calls += 1
            entered.set()
            assert release.wait(5)
    pool = ProcessPool()
    monkeypatch.setattr(kernel_plan, "_state", {"compiler_pool": pool})
    kernel_plan.close()
    assert entered.wait(5)
    joined = threading.Thread(target=lambda: kernel_plan.close(wait=True))
    joined.start()
    release.set()
    joined.join(5)
    assert not joined.is_alive()
    assert pool.calls == 1
    kernel_plan.close(wait=True)
    assert pool.calls == 1


@pytest.mark.parametrize("covered", [True, False])
def test_aot_hint_precedes_warm_restore_and_keeps_ordinary_funnel(monkeypatch, plan_dir, covered):
    requests = [(("A", ()), {}), (("B", ()), {})]
    _record_run(monkeypatch, plan_dir, _StubCompiler(), requests)
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    restored = []

    class Images:
        def get(self, arguments):
            restored.append(arguments["source"])
            # B has a retained image; A uses the ordinary compiler stub.
            return (b"retained-B", None) if arguments["source"] == "B" else None
        def put(self, arguments, result):
            pass
        def close(self):
            pass
        def receipt(self):
            return {}

    monkeypatch.setattr(kernel_plan, "_warm_images", lambda: Images())
    monkeypatch.setattr(kernel_plan, "_aot_coverage", lambda: {"A"} if covered else set())
    monkeypatch.setattr(kernel_plan, "_aot_cached", lambda arguments, arch, held: arguments["source"] in held)
    record = kernel_plan.start()
    kernel_plan.close(wait=True)
    assert restored == (["B"] if covered else ["A", "B"])
    assert record["pool_skipped_aot"] == int(covered)
    assert record["pool_skipped_images"] == 1
    assert record["pool_compiled"] == int(not covered)
    assert record["entries"] == 1  # Preserve receipt meaning: non-image rows.
    # A synthetic coverage hint leaves the ordinary compiler funnel callable.
    result = here._compile_using_nvrtc_no_warning("A", ())
    assert result == _StubCompiler()._compile_using_nvrtc_no_warning("A", ())
    assert len(here.calls) == 1


def test_parallel_member_initialization_installs_one_compile_funnel(monkeypatch, plan_dir):
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    entered = threading.Event()
    second_attempted = threading.Event()
    identities = []

    def identity():
        identities.append(1)
        entered.set()
        assert second_attempted.wait(2)
        # Let the competing caller reach install while this installation is open.
        time.sleep(0.05)
        return {"stub": 1}

    monkeypatch.setattr(kernel_plan, "compiler_identity", identity)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(kernel_plan.install, plan_dir, "ensemble")
        assert entered.wait(2)
        def second():
            second_attempted.set()
            return kernel_plan.install(plan_dir, "ensemble")
        other = pool.submit(second)
        assert first.result(timeout=3)
        assert other.result(timeout=3)
    assert len(identities) == 1
    result = compiler._compile_using_nvrtc_no_warning("headline")
    assert result[0].startswith(b"image:headline:")
    assert len(compiler.calls) == 1


def test_an_idle_compile_worker_does_not_hold_the_exit():
    """The hl-3 headline run (2026-10-24): a finished hex forecast sat in multiprocessing's exit join
    on one idle spawn worker of this pool. The exit drop terminates the workers, even when the
    pool's shutdown raises."""
    import concurrent.futures
    import multiprocessing
    import time
    from woof import kernel_plan
    pool = concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    processes = pool._processes
    pool.submit(time.sleep, 0).result(60)   # the worker is up and idle

    class Raising:
        def shutdown(self, **_kwargs):
            raise OSError("wakeup pipe closed")

    kernel_plan._drop_at_exit(Raising(), processes)
    for process in list(processes.values()):
        process.join(10)
        assert not process.is_alive()
    pool.shutdown(wait=True, cancel_futures=True)


#: Run in a spawn compile worker by the builtin ``exec``: write a result header that promises 1 MiB, send
#: 1 KiB of it, then wait to be stopped -- a worker caught by the exit drop in the middle of sending its image.
_HALF_SENT_RESULT = """
import os, struct, sys, time
frame = sys._getframe(0)
while frame is not None and "result_queue" not in frame.f_locals:
    frame = frame.f_back
os.write(frame.f_locals["result_queue"]._writer.fileno(), struct.pack("!i", 1 << 20) + bytes(1024))
time.sleep(600)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="the half-sent result is written with os.write on the POSIX "
                    "result pipe; on Windows that pipe is a named-pipe handle with no file descriptor")
def test_a_worker_stopped_mid_result_does_not_hold_the_exit():
    """woof hex fix/hex-exit-hang 8137639c (RRFS restart proofs, 2026-10-29), this file's twin: finished
    forecasts never exited and held their card lease. A compile worker the exit drop stopped while it was sending
    its image left the pool's manager thread reading the rest forever, and concurrent.futures' exit hook joins
    that thread with no timeout."""
    import multiprocessing
    pool = concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    processes, results = pool._processes, pool._result_queue
    pool.submit(exec, _HALF_SENT_RESULT)
    manager = pool._executor_manager_thread
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:   # until the manager is inside the read of the half-sent result
        try:
            if "pipe_read" in open(f"/proc/self/task/{manager.native_id}/wchan").read():
                break
        except OSError:
            time.sleep(3)
            break
        time.sleep(0.05)
    pool.shutdown(wait=False)  # what kernel_plan.close() does at the end of the forecast
    try:
        kernel_plan._drop_at_exit(pool, processes, results)
        manager.join(20)
        assert not manager.is_alive()
    finally:
        results._writer.close()  # a red run must not also hang the test session's own exit
        manager.join(20)


#: A run in miniature: it opens the compile pool from a thread that ends right after submitting (the
#: funnel and ahead() submit from any thread), checks its two compiling workers outlived that thread,
#: prints their pids, then ends without any exit hook (``exit``: os._exit(0)) or waits to be SIGKILLed.
_POOL_OPENER = """
import os, sys, threading, time
from woof import kernel_plan
pools = []
def setup():
    pool = kernel_plan.compiler_pool(2)
    pools.append(pool)
    for _ in range(2):
        pool.submit(time.sleep, 600)
thread = threading.Thread(target=setup)
thread.start()
thread.join()
pool = pools[0]
deadline = time.monotonic() + 60
while len(pool._processes) < 2 and time.monotonic() < deadline:
    time.sleep(0.05)
time.sleep(1.0)
print(" ".join(str(pid) for pid, process in pool._processes.items() if process.is_alive()), flush=True)
if sys.argv[1] == "exit":
    os._exit(0)
time.sleep(600)
"""


def _pid_running(pid):
    """A pid that exists and is not a zombie (a dead worker waiting for init to reap it is gone)."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="worker liveness is read from /proc (Linux)")
@pytest.mark.parametrize("end", ["exit", "kill"])
def test_compile_workers_end_with_the_process_that_opened_them(end):
    """2026-10-29 08:17-08:21Z (woof hex fix/hex-compile-worker-lifetime, this file's twin): spawn
    compile workers of runs that had ended (a WOOF prewarm stopped by a signal, hex forecasts OOM-killed)
    held 62.5 GB until killed by PID; the OOM killer then took release containers and lanes. The workers must
    also outlive the short-lived thread that submitted to them."""
    import os
    import signal
    import subprocess
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    opener = subprocess.Popen([sys.executable, "-c", _POOL_OPENER, end], stdout=subprocess.PIPE, text=True,
                              env=environment)
    workers = []
    try:
        workers = [int(pid) for pid in opener.stdout.readline().split()]
        assert len(workers) == 2, "a compile worker died with the thread that submitted to it"
        if end == "kill":
            opener.send_signal(signal.SIGKILL)
        opener.wait(60)
        deadline = time.monotonic() + 10
        while any(_pid_running(pid) for pid in workers) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not [pid for pid in workers if _pid_running(pid)]
    finally:
        if opener.poll() is None:
            opener.kill()
            opener.wait(10)
        for pid in workers:
            if _pid_running(pid):
                os.kill(pid, signal.SIGKILL)   # a red run leaves no orphan behind
