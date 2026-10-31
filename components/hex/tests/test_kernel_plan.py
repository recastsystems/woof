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

import pytest

from woof.hex import kernel_plan


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
        return (f"image:{source}:{options}:{resolved}".encode(), {"k": "mangled"} if name_expressions else None)

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
    monkeypatch.setattr(kernel_plan, "_raw_flight", {})
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
    _record_run(monkeypatch, plan_dir, _StubCompiler(arch="120"), [(("A", ("-x",), "120", "h120.cu"), {})])
    here = _StubCompiler(arch="89")
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    assert here.calls[0][2] == "89"
    # CuPy names its program after an architecture-bound hash; the image does not depend on it.
    here._compile_using_nvrtc_no_warning("A", ("-x",), "89", "h89.cu")
    assert len(here.calls) == 1


def test_program_name_is_matched_when_it_can_change_the_image(monkeypatch, plan_dir):
    source = 'extern "C" __global__ void k(int* a){ assert(a); }'
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [((source, ("-x",), None, "one.cu"), {})])
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    here._compile_using_nvrtc_no_warning(source, ("-x",), None, "two.cu")
    assert len(here.calls) == 2  # assert embeds the file name: not served under another name
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


@pytest.mark.parametrize("damaged", [False, True])
def test_an_unknown_configuration_does_not_scan_or_compile_other_plans(monkeypatch, plan_dir, damaged):
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("A", ()), {})], tag="one")
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("B", ()), {})], tag="two")
    here = _StubCompiler()
    _fresh(monkeypatch, here)
    kernel_plan.install(plan_dir=plan_dir, tag_text="three")
    if damaged:
        kernel_plan._state["plan"].write_text("{invalid json\n")
    reads = []
    original_read = kernel_plan._read

    def read(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(kernel_plan, "_read", read)
    record = kernel_plan.start()
    kernel_plan.close()
    assert reads == ([kernel_plan._state["plan"]] if damaged else [])
    assert record["plan_source"] == "none" and record["entries"] == 0
    assert record["pool_compiled"] == 0 and not here.calls
    # Actual requests still compile through the ordinary funnel and retain
    # their image and mapping, even when this configuration has no plan yet.
    arguments = ("C", ("--fmad=false",), None, "c.cu", ["k"])
    assert here._compile_using_nvrtc_no_warning(*arguments) == (
        _StubCompiler()._compile_using_nvrtc_no_warning(*arguments))
    assert len(here.calls) == 1


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


def test_tag_binds_shape_options_and_numerics_but_ignores_output(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_NUMERICS", "fast")
    args = ["forecast", "--mesh", "x1.163842", "--levels", "55", "--out", "/a"]
    tag = kernel_plan.default_tag_text(args)
    assert tag == kernel_plan.default_tag_text(args[:-1] + ["/b"])
    assert tag == kernel_plan.default_tag_text(args + ["--scratch", "/run-1/scratch"])
    assert tag == kernel_plan.default_tag_text(args + ["--scratch=/run-2/scratch"])
    assert tag != kernel_plan.default_tag_text(["forecast", "--mesh", "x1.40962", "--levels", "55"])
    assert tag != kernel_plan.default_tag_text(["forecast", "--mesh", "x1.163842", "--levels", "49"])
    monkeypatch.setenv("WOOF_HEX_NUMERICS", "strict")
    assert tag != kernel_plan.default_tag_text(args)


def test_members_of_one_configuration_share_one_tag(monkeypatch):
    # Breakage: each DA member's own --init/--iau-analysis/--start-time made a new ~22 MB plan
    # and manifest that no other member could find (own24 cycle members, 2026-10-28).
    monkeypatch.setenv("WOOF_HEX_NUMERICS", "fast")
    def member(n, start):
        return ["forecast", "--mesh", "hex-vr-1", "--init", f"/c/{start}/mem{n:03d}/composed.nc",
                "--start-time", start, "--init-source", f"ensemble member {n:03d}",
                "--iau-analysis", f"/c/{start}/mem{n:03d}/analysis.nc", "--iau-window-seconds", "600"]
    tag = kernel_plan.default_tag_text(member(1, "2023-07-09_19:00:00"))
    assert tag == kernel_plan.default_tag_text(member(2, "2023-07-09_20:00:00"))
    assert tag == kernel_plan.default_tag_text(
        [word.replace("--start-time", "--start-time=2023-07-10_00:00:00") for word in member(3, "x")
         if word != "x"])
    # The flags still bind: a run without IAU or with another window is another configuration.
    plain = member(1, "2023-07-09_19:00:00")[:-4]
    assert tag != kernel_plan.default_tag_text(plain)
    assert tag != kernel_plan.default_tag_text(member(1, "2023-07-09_19:00:00")[:-1] + ["900"])


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


def test_save_serializes_an_independent_snapshot(monkeypatch, plan_dir):
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text='snapshot')
    with kernel_plan.cache_entry('host-test', '/cache/original.cubin'):
        compiler._compile_using_nvrtc_no_warning('snapshot source')
    row = next(iter(kernel_plan._requested.values()))
    assert row['cache_entries'] == {'89':'/cache/original.cubin'}
    original_is_file = kernel_plan.Path.is_file
    def advance_compile_metadata(path):
        # A compile finishes after save released its lock, before serialization.
        row['cache_entries']['120'] = '/cache/new.cubin'
        return original_is_file(path)
    monkeypatch.setattr(kernel_plan.Path, 'is_file', advance_compile_metadata)
    kernel_plan.save()
    stored = json.loads(kernel_plan._state['plan'].read_text())['entries'][0]
    assert stored['cache_entries'] == {'89':'/cache/original.cubin'}
    assert row['cache_entries']['120'] == '/cache/new.cubin'
    json.dumps(kernel_plan.receipt(), allow_nan=False)


def test_pool_is_fully_fed_before_any_compiler_call(monkeypatch, plan_dir):
    requests = [((f"kernel-{i}", ()), {}) for i in range(12)]
    _record_run(monkeypatch, plan_dir, _StubCompiler(), requests)
    here = _StubCompiler()
    original = here._compile_using_nvrtc_no_warning
    def compile_(*args, **kwargs):
        assert len(kernel_plan._flight) == len(requests)
        return original(*args, **kwargs)
    here._compile_using_nvrtc_no_warning = compile_
    _fresh(monkeypatch, here)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: sum(r["entries"] for r in record["pool_threads"]) == len(requests))
    assert record["workers"] == kernel_plan._workers()
    assert record["pool_failed"] == 0 and len(here.calls) == len(requests)
    assert record["pool_cpu_seconds"] >= 0


def test_aot_coverage_is_only_an_exact_compile_scheduling_hint(monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof.hex.cuda_backend.aot import source_hash
    monkeypatch.setitem(sys.modules, "woof.numerics", SimpleNamespace(effective_options=tuple))
    source = 'extern "C" __global__ void k(){}'
    row = dict(source_sha256=source_hash(source), options=["--std=c++17"], arch="sm_120")
    held = kernel_plan._aot_coverage(SimpleNamespace(reason=None, rows=[row]))
    arguments = dict(source=source, options=["--std=c++17", "-ftz=true"], arch="device",
                     name_expressions=None, method=None)
    assert kernel_plan._aot_cached(arguments, "120", held)
    for changed in (dict(options=["--std=c++17"]), dict(arch="89"),
                    dict(name_expressions=["k"]), dict(source=source+"//changed"),
                    dict(method="ptx")):
        assert not kernel_plan._aot_cached({**arguments, **changed}, "120", held)
    assert not kernel_plan._aot_coverage(SimpleNamespace(reason="compiler_mismatch", rows=[row]))
    direct = {**row, "compiler_route": "direct-nvrtc"}
    assert not kernel_plan._aot_cached(arguments, "120", kernel_plan._aot_coverage(SimpleNamespace(reason=None, rows=[direct])))


def test_aot_scheduling_hint_binds_assertion_program_filename(monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof.hex.cuda_backend.aot import source_hash
    monkeypatch.setitem(sys.modules, "woof.numerics", SimpleNamespace(effective_options=tuple))
    source = 'extern "C" __global__ void k(){assert(0);}'
    row = dict(source_sha256=source_hash(source), options=[], arch="sm_120",
               compiler_route="direct-nvrtc", compile_request=dict(filename="bound.cu"))
    held = kernel_plan._aot_coverage(SimpleNamespace(reason=None, rows=[row]))
    arguments = dict(source=source, options=[], arch="device", name_expressions=None,
                     method=None, filename="bound.cu")
    assert kernel_plan._aot_cached(arguments, "120", held)
    assert not kernel_plan._aot_cached({**arguments, "filename": "other.cu"}, "120", held)



def test_foreground_compile_during_background_plan_loading_is_not_duplicated(monkeypatch, plan_dir):
    from woof.hex import kernel_prewarm
    monkeypatch.delenv(kernel_prewarm.ENVIRONMENT, raising=False)
    requests = [(("foreground-first", ("--fmad=false",), "89"), {})]
    _record_run(monkeypatch, plan_dir, _StubCompiler(), requests)
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    entered, release = threading.Event(), threading.Event()
    original = kernel_plan.load_plan
    def held():
        entered.set()
        assert release.wait(5)
        return original()
    monkeypatch.setattr(kernel_plan, "load_plan", held)
    monkeypatch.setattr(kernel_prewarm, "enabled", lambda: True)
    monkeypatch.setattr(kernel_prewarm, "_state", {"recording": False, "replay": None})
    monkeypatch.setattr(kernel_prewarm, "_start_manifest_replay", lambda device, pool: None)
    future = kernel_prewarm.start_replay_async()
    try:
        assert entered.wait(5)
        image = compiler._compile_using_nvrtc_no_warning(*requests[0][0])
        assert len(compiler.calls) == 1
    finally:
        release.set()
    future.result(5)
    assert compiler._compile_using_nvrtc_no_warning(*requests[0][0]) == image
    assert len(compiler.calls) == 1
    assert kernel_plan.receipt()["replay"]["pool_compiled"] == 0


@pytest.mark.parametrize("pack_valid", [True, False])
def test_aot_covered_plan_does_not_bind_unused_warm_headers(monkeypatch, plan_dir, pack_valid):
    from types import SimpleNamespace
    from woof.hex import kernel_warm
    from woof.hex.cuda_backend import aot
    requests = [((source, (), "89"), {}) for source in ("packed", "warm", "missing")]
    _record_run(monkeypatch, plan_dir, _StubCompiler(), requests)
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    compiler.__name__ = "cupy.cuda.compiler"
    monkeypatch.setattr(kernel_plan, "_process_compiler_identity", lambda arch: None)
    monkeypatch.setattr(aot, "effective_options", tuple)
    row = dict(source_sha256=aot.source_hash("packed"), options=[], arch="sm_89",
               compiler_route="direct-nvrtc", compile_request=dict(filename="kern.cu"))
    monkeypatch.setattr(aot, "Pack", lambda: SimpleNamespace(
        reason=None if pack_valid else "compiler_mismatch", rows=[row], _architectures={"sm_89"}))
    monkeypatch.setattr(kernel_plan, "_catalog_rows", lambda arch: [])
    restored = []
    def restore(arguments, arch):
        restored.append(arguments["source"])
        return (b"warm-image", None) if arguments["source"] == "warm" else None
    monkeypatch.setattr(kernel_warm, "restore", restore)
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == (1 if pack_valid else 2))
    assert set(restored) == ({"warm", "missing"} if pack_valid else {"packed", "warm", "missing"})
    assert record["pool_skipped_aot"] == int(pack_valid)
    assert record["pool_restored_warm"] == 1
    assert {call[0] for call in compiler.calls} == ({"missing"} if pack_valid else {"packed", "missing"})
    assert compiler._compile_using_nvrtc_no_warning("warm", (), "89") == (b"warm-image", None)
    if pack_valid:
        # If the later AOT loader declines the hint, the ordinary funnel is
        # still available. Scheduling never installs an unchecked image.
        assert not any(call[0] == "packed" for call in compiler.calls)
        image = compiler._compile_using_nvrtc_no_warning("packed", (), "89")
        assert image == _StubCompiler()._compile_using_nvrtc_no_warning("packed", (), "89")
        assert sum(call[0] == "packed" for call in compiler.calls) == 1


@pytest.mark.parametrize("rejection", ["compile_request_mismatch", "cubin_digest_mismatch"])
def test_real_aot_rejection_after_plan_hint_reaches_compiler(monkeypatch, plan_dir, rejection):
    """A scheduling hint cannot bypass the real loader's request or image checks."""
    import base64
    import hashlib
    import sys
    from types import SimpleNamespace
    from woof.hex import kernel_pack, kernel_warm
    from woof.hex.cuda_backend import aot

    monkeypatch.delenv("WOOF_HEX_AOT", raising=False)
    request = (("packed", (), "89", "named.cu"), {"cache_in_memory": True})
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [request])
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    assert kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh")
    compiler.__name__ = "cupy.cuda.compiler"
    monkeypatch.setattr(kernel_plan, "_process_compiler_identity", lambda arch: None)
    monkeypatch.setattr(aot, "effective_options", tuple)
    image = b"\x7fELFtest-image"
    key = "nvrtc.direct." + aot.source_hash("packed")
    row = dict(module_key=key, source_sha256=aot.source_hash("packed"),
               options=[], arch="sm_89", compiler_route="direct-nvrtc",
               compile_request=aot.direct_metadata("named.cu", rejection != "compile_request_mismatch"),
               cubin_base64=base64.b64encode(image).decode(),
               cubin_sha256=hashlib.sha256(image).hexdigest())
    if rejection == "cubin_digest_mismatch":
        row["cubin_sha256"] = "0" * 64
    identity = {"fixture": "loader-rejection"}
    path = plan_dir.parent / "rejected-pack.json"
    kernel_pack.write(path, identity, [row])
    selected = aot.Pack(path, identity)
    assert selected.reason is None
    monkeypatch.setattr(aot, "Pack", lambda: selected)
    def unexpected_restore(*args):
        raise AssertionError("AOT-covered request must skip warm header binding")
    monkeypatch.setattr(kernel_warm, "restore", unexpected_restore)
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["pool_skipped_aot"] == 1
    assert record["pool_compiled"] == record["pool_restored_warm"] == 0
    assert compiler.calls == []

    def unused_cached_route(*args, **kwargs):
        raise AssertionError("direct request entered cached compiler route")
    compiler._compile_with_cache_cuda = unused_cached_route
    compiler.compile_using_nvrtc = lambda *args, **kwargs: compiler._compile_using_nvrtc_no_warning(*args, **kwargs)
    monkeypatch.setitem(sys.modules, "cupy.cuda", SimpleNamespace(compiler=compiler))
    marks = []
    monkeypatch.setattr(aot, "mark", lambda source, key, seconds, **kwargs: marks.append((source, kwargs.get("reason"))))
    aot._bridge(None, selected)
    expected = _StubCompiler()._compile_using_nvrtc_no_warning(*request[0], **request[1])
    assert compiler.compile_using_nvrtc(*request[0], **request[1]) == expected
    assert compiler.compile_using_nvrtc(*request[0], **request[1]) == expected
    assert len(compiler.calls) == 1  # Real fallback joins the plan's single-flight cache.
    assert marks == [("jit", rejection), ("jit", rejection)]

    # Matching metadata and intact image exercise the positive loader path too.
    row["compile_request"] = aot.direct_metadata("named.cu", True)
    row["cubin_sha256"] = hashlib.sha256(image).hexdigest()
    valid_path = plan_dir.parent / "valid-pack.json"
    kernel_pack.write(valid_path, identity, [row])
    valid = selected.__class__(valid_path, identity)
    selected.rows = valid.rows
    selected._by_key = valid._by_key
    assert compiler.compile_using_nvrtc(*request[0], **request[1]) == (image, None)
    assert len(compiler.calls) == 1
    assert marks[-1] == ("aot", None)


@pytest.mark.parametrize("owned", [True, False])
def test_existing_owner_checked_before_warm_wrapper(monkeypatch, plan_dir, owned):
    """Use the real warm installer with a CPU compiler/device adapter."""
    import sys
    from types import SimpleNamespace
    from woof.hex import kernel_warm
    from woof.hex.cuda_backend import aot
    from woof.certify import compile_platform

    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    compiler.__name__ = "cupy.cuda.compiler"
    original = compiler._compile_using_nvrtc_no_warning
    def prior(*args, **kwargs):
        return original(*args, **kwargs)
    if owned:
        setattr(prior, kernel_plan._MARK, True)
    compiler._compile_using_nvrtc_no_warning = prior
    preprocess = lambda *args, **kwargs: b"preprocess"
    compiler._preprocess = preprocess
    banner = lambda: "test-banner"
    monkeypatch.setattr(compile_platform, "nvrtc_banner", banner)
    monkeypatch.setitem(sys.modules, "cupy.cuda", SimpleNamespace(compiler=compiler))
    monkeypatch.setattr(aot, "compiler_identity", lambda: {"nvrtc": [13, 3]})
    monkeypatch.setattr(kernel_warm, "_STATE", None)
    monkeypatch.setattr(kernel_warm, "_root", lambda: plan_dir / "warm")
    try:
        assert kernel_plan.install(plan_dir=plan_dir, tag_text="owner-order") is (not owned)
        if owned:
            assert compiler._compile_using_nvrtc_no_warning is prior
            assert compiler._preprocess is preprocess
            assert compile_platform.nvrtc_banner is banner
            assert kernel_warm._STATE is None
            assert not kernel_plan._state["installed"]
        else:
            funnel = compiler._compile_using_nvrtc_no_warning
            assert getattr(funnel, kernel_plan._MARK, False)
            assert funnel.__wrapped__.__wrapped__ is prior
            assert compiler._preprocess.__wrapped__ is preprocess
            assert compile_platform.nvrtc_banner.__wrapped__ is banner
            assert kernel_warm._STATE is not None
            assert kernel_plan._state["installed"]
        assert compiler.calls == []
        assert not plan_dir.exists()
    finally:
        if kernel_warm._STATE is not None:
            kernel_warm._STATE.pool.shutdown(wait=True)


def test_incomplete_plan_is_completed_from_catalog_for_unpacked_compiler(monkeypatch, plan_dir):
    from types import SimpleNamespace
    from woof.hex import kernel_warm
    from woof.hex.cuda_backend import aot
    _record_run(monkeypatch, plan_dir, _StubCompiler(), [(("prefix", (), "89"), {})])
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    kernel_plan.install(plan_dir=plan_dir, tag_text='forecast --mesh')
    compiler.__name__ = 'cupy.cuda.compiler'
    monkeypatch.setattr(aot, 'Pack', lambda: SimpleNamespace(reason='compiler_mismatch', rows=[]))
    monkeypatch.setattr(aot, 'effective_options', tuple)
    monkeypatch.setattr(kernel_plan, '_process_compiler_identity', lambda arch: None)
    monkeypatch.setattr(kernel_warm, 'restore', lambda arguments, arch: None)
    args = kernel_plan._arguments('later', (), '89', 'kern.cu', None, False, None)
    row = dict(sha256=kernel_plan._key(args), arguments=args, seconds=None)
    monkeypatch.setattr(kernel_plan, '_catalog_rows', lambda arch: [row])
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record['pool_compiled'] == 2)
    assert {call[0] for call in compiler.calls} == {'prefix', 'later'}
    assert record['plan_source'].endswith('+catalog')


def _raw_catalog(monkeypatch, plan_dir, raw_rows, funnel_rows=()):
    """A cold start with no usable pack whose catalog holds ``raw_rows`` (thread executor)."""
    from types import SimpleNamespace
    from woof.hex import kernel_warm
    from woof.hex.cuda_backend import aot
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    kernel_plan.install(plan_dir=plan_dir, tag_text="forecast --mesh new")
    compiler.__name__ = "cupy.cuda.compiler"
    monkeypatch.setattr(aot, "Pack", lambda: SimpleNamespace(reason="pack_missing", rows=[]))
    monkeypatch.setattr(aot, "effective_options", tuple)
    monkeypatch.setattr(kernel_plan, "_process_compiler_identity", lambda arch: None)
    monkeypatch.setattr(kernel_warm, "restore", lambda arguments, arch: None)
    monkeypatch.setattr(kernel_plan, "_catalog_rows", lambda arch: [*funnel_rows, *raw_rows])
    return compiler


def _raw_row(source="gf-source", options=("--fmad=false", "-DGF=1"), filename="gf_mpas_v842.cu"):
    arguments = {"source": source, "options": list(options), "arch": "device", "filename": filename,
                 "cache_in_memory": False, "name_expressions": None, "method": None}
    return {"sha256": kernel_plan._raw_key(source, options, "device", filename), "arguments": arguments,
            "seconds": None, "route": kernel_plan.RAW_ROUTE, "physics_family": "gf_mpas"}


def test_a_cold_start_compiles_raw_nvrtc_rows_on_the_pool_and_the_unit_takes_that_image(monkeypatch, plan_dir):
    """THE BREAKAGE (an RTX 5090 host and V8X6000, 2026-10-29): _catalog_rows skipped raw NVRTC rows, so with no pack
    Grell-Freitas compiled in the foreground on one core at its first call while the card waited."""
    from woof.hex import kernel_compile_worker, kernel_warm
    calls = []

    def raw_compile(arguments, arch):
        calls.append((arguments["source"], tuple(arguments["options"]), arch, arguments["filename"],
                      threading.current_thread().name))
        return b"\x7fELF-gf-" + arch.encode()
    monkeypatch.setattr(kernel_compile_worker, "raw_compile", raw_compile)
    monkeypatch.setattr(kernel_warm, "raw_cached", lambda *args: False)
    _raw_catalog(monkeypatch, plan_dir, [_raw_row()])
    record = kernel_plan.start()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    assert record["pool_raw"] == 1 and record["entries"] == 1
    # The unit's own request (aot.raw_image passes the card's arch, not "device") is served the pool's image.
    image = kernel_plan.raw_planned("gf-source", ("--fmad=false", "-DGF=1"), "89", "gf_mpas_v842.cu")
    assert image == b"\x7fELF-gf-89"
    assert [call[:4] for call in calls] == [("gf-source", ("--fmad=false", "-DGF=1"), "89", "gf_mpas_v842.cu")]
    assert kernel_plan.POOL_PREFIX in calls[0][4]
    assert kernel_plan.receipt()["calls"]["raw_served"] == 1
    # A request the plan never held (other options, other program name) goes to the unit's own path.
    assert kernel_plan.raw_planned("gf-source", ("--fmad=true", "-DGF=1"), "89", "gf_mpas_v842.cu") is None
    assert kernel_plan.raw_planned("gf-source", ("--fmad=false", "-DGF=1"), "89", "other.cu") is None


def test_a_raw_row_still_queued_is_taken_back_by_the_unit(monkeypatch, plan_dir):
    from woof.hex import kernel_compile_worker, kernel_warm
    monkeypatch.setenv(kernel_plan.WORKERS_ENVIRONMENT, "1")
    calls = []
    monkeypatch.setattr(kernel_compile_worker, "raw_compile", lambda arguments, arch: calls.append(arch) or b"x")
    monkeypatch.setattr(kernel_warm, "raw_cached", lambda *args: False)
    _fresh(monkeypatch, _StubCompiler())  # arguments relative to the stub card, so no GPU is needed
    slow = kernel_plan._arguments("slow", (), "89", "kern.cu", None, False, None)
    compiler = _raw_catalog(monkeypatch, plan_dir, [_raw_row()],
                            [dict(sha256=kernel_plan._key(slow), arguments=slow, seconds=9.0)])
    compiler.gate = threading.Event()
    record = kernel_plan.start()
    # The one worker holds the slow funnel row (longest first); the raw row waits behind it.
    assert kernel_plan.raw_planned("gf-source", ("--fmad=false", "-DGF=1"), "89", "gf_mpas_v842.cu") is None
    compiler.gate.set()
    kernel_plan.close()
    _wait(lambda: record["pool_compiled"] == 1)
    assert calls == [] and kernel_plan.receipt()["calls"]["raw_taken_back"] == 1


def test_a_raw_row_the_warm_store_holds_is_not_compiled_again(monkeypatch, plan_dir):
    from woof.hex import kernel_compile_worker, kernel_warm
    monkeypatch.setattr(kernel_compile_worker, "raw_compile",
                        lambda arguments, arch: pytest.fail("a held raw image compiled again"))
    seen = []
    monkeypatch.setattr(kernel_warm, "raw_cached", lambda *args: seen.append(args) or True)
    _raw_catalog(monkeypatch, plan_dir, [_raw_row()])
    record = kernel_plan.start()
    kernel_plan.close()
    assert record["pool_raw"] == 0 and record["pool_restored_warm"] == 1
    assert seen == [("gf-source", ["--fmad=false", "-DGF=1"], "89", "gf_mpas_v842.cu")]
    assert kernel_plan.raw_planned("gf-source", ("--fmad=false", "-DGF=1"), "89", "gf_mpas_v842.cu") is None


def test_catalog_rows_carry_raw_nvrtc_units_with_their_exact_options(monkeypatch):
    from types import SimpleNamespace
    from woof.hex.cuda_backend import aot, aot_build
    compiler = SimpleNamespace(_get_bool_env_variable=lambda name, default: True)  # debug flags never reach raw
    monkeypatch.setattr(kernel_plan, "_compiler", lambda: compiler)
    monkeypatch.setattr(aot, "compiler_identity", lambda: {"nvrtc": [13, 4]})
    raw = {"source": "gf", "options": ["--fmad=false", "-I@SITE@/x"], "compiler_route": "raw-nvrtc",
           "compile_request": {"route": "raw-nvrtc", "filename": "gf_mpas_v842.cu"}, "physics_family": "gf_mpas"}
    monkeypatch.setattr(aot_build, "inventory", lambda **kwargs: [raw, dict(raw)])
    monkeypatch.setattr(aot, "local_options", lambda options: [o.replace("@SITE@", "/site") for o in options])
    rows = kernel_plan._catalog_rows("120")
    assert len(rows) == 1
    (row,) = rows
    assert row["route"] == kernel_plan.RAW_ROUTE and row["physics_family"] == "gf_mpas"
    assert row["arguments"]["options"] == ["--fmad=false", "-I/site/x"]
    assert row["arguments"]["filename"] == "gf_mpas_v842.cu" and row["arguments"]["arch"] == "device"
    assert row["sha256"] == kernel_plan._raw_key("gf", ["--fmad=false", "-I/site/x"], "device", "gf_mpas_v842.cu")


def test_raw_image_takes_the_pool_image_without_compiling(monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof.hex import kernel_warm
    from woof.hex.cuda_backend import aot
    monkeypatch.setattr(kernel_warm, "_STATE", None)
    monkeypatch.setattr(kernel_warm, "install", lambda **_: None)
    monkeypatch.setattr(aot, "Pack", lambda: pytest.fail("the pool image needs no pack lookup"))
    monkeypatch.setitem(sys.modules, "cupy_backends.cuda.libs",
                        SimpleNamespace(nvrtc=SimpleNamespace(createProgram=lambda *a: pytest.fail("compiled"))))
    taken = []
    monkeypatch.setattr(kernel_plan, "raw_planned", lambda *args: taken.append(args) or b"\x7fELFpool")
    assert aot.raw_image("src", ("--fmad=false",), "sm_120", filename="gf.cu") == b"\x7fELFpool"
    assert taken == [("src", ("--fmad=false",), "120", "gf.cu")]


@pytest.mark.parametrize("table", ["mpasphys_v842", "mpasphys_rrfs_44c0849"])
def test_a_v842_run_leaves_the_rrfs_forks_catalog_rows_out(monkeypatch, plan_dir, table):
    """THE BREAKAGE (an RTX 5090 host, 2026-10-29): a cold strict hl-3 (v8.4.2 convection_permitting) compiled the RRFS
    fork's rows too; two workers held 35-37 GB of host RAM each.  Slots are not matched to families for the
    skip: thompson_aero runs thompson_workspace and sf_mynn runs sfclay_mm5 (V8X6000, 2026-10-29)."""
    from types import SimpleNamespace
    from woof.hex import kernel_compile_worker, kernel_warm
    monkeypatch.setattr(kernel_compile_worker, "raw_compile", lambda arguments, arch: b"gf")
    monkeypatch.setattr(kernel_warm, "raw_cached", lambda *args: False)
    _fresh(monkeypatch, _StubCompiler())  # arguments relative to the stub card, so no GPU is needed

    def physics(source, family):
        arguments = kernel_plan._arguments(source, (), "89", "kern.cu", None, False, None)
        return dict(sha256=kernel_plan._key(arguments), arguments=arguments, seconds=None, physics_family=family)
    rows = [physics("thompson", "thompson_workspace"), physics("sfclay", "sfclay_mm5"),
            physics("edmf", "bl_mynnedmf"), physics("tempo", "tempo_rrfs_phase"),
            physics("rrfs-mm5", "rrfs_sfclay_mm5"), physics("dycore", "")]
    compiler = _raw_catalog(monkeypatch, plan_dir, [_raw_row()], rows)
    config = SimpleNamespace(table_key=table, active=lambda slot: True,
                             selectors={"microphysics": "mp_thompson_aerosols", "surface_layer": "sf_mynn"})
    if table != "mpasphys_v842":
        config.selectors = {"microphysics": "mp_tempo", "surface_layer": "sf_mynn"}
    kernel_plan.prioritize_physics(config)
    record = kernel_plan.start()
    kernel_plan.close()
    kept = ["dycore", "sfclay", "thompson"] + (["edmf", "rrfs-mm5", "tempo"] if table != "mpasphys_v842" else [])
    _wait(lambda: record["pool_compiled"] == len(kept) + 1)
    assert sorted(call[0] for call in compiler.calls) == sorted(kept)
    assert record["pool_skipped_rrfs_only"] == (3 if table == "mpasphys_v842" else 0)


def test_the_catalog_starts_the_longest_sources_first():
    rows = [dict(id="small", arguments=dict(source="x" * 10)), dict(id="noah", arguments=dict(source="x" * 9000)),
            dict(id="timed", arguments=dict(source="x"), seconds=3.0)]
    assert [row["id"] for row in sorted(rows, key=kernel_plan._compile_priority)] == ["timed", "noah", "small"]
