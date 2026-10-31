"""First-shape jobs join the owning Hex compiler without recording warm-up side effects."""
from concurrent.futures import Future
from types import SimpleNamespace
import os
import threading

import pytest

from woof.hex import kernel_plan, kernel_prewarm


def test_first_shape_entry_point_exists():
    assert callable(getattr(kernel_prewarm, "start_static", None))


def test_selected_backend_constructors_dispatch_static_jobs():
    import ast
    from pathlib import Path
    root = Path(kernel_prewarm.__file__).parent
    for filename, cls in (("cuda_arwen_physics_v841.py", "PersistentTwoPhaseCudaPhysicsBackendV841"),
                          ("cuda_suite_physics_v841.py", "PersistentSuiteCudaPhysicsBackendV841")):
        module = ast.parse((root / filename).read_text())
        node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls)
        init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "start_static"]
        assert len(calls) == 1
        assert ast.unparse(calls[0].args[0]) == "self._seam._cfg"


@pytest.fixture
def static_setup(monkeypatch, tmp_path):
    import sys
    from woof.hex import kernel_static
    from test_kernel_plan import _fresh, _StubCompiler
    compiler = _StubCompiler()
    _fresh(monkeypatch, compiler)
    monkeypatch.setenv("WOOF_KERNEL_PLAN_WORKERS", "2")
    monkeypatch.setattr(kernel_prewarm, "enabled", lambda: True)
    monkeypatch.setattr(kernel_static, "_runs", {})
    monkeypatch.setattr(kernel_static, "_tag", lambda run: "selected-shape")
    monkeypatch.setattr(kernel_plan, "_capture", threading.local())
    monkeypatch.setattr(kernel_plan, "_process_compiler_identity", lambda arch: None)
    class Sentinel(BaseException):
        def __init__(self):
            self.key = None
    engine = SimpleNamespace(_capture=threading.local(), _Captured=Sentinel)
    monkeypatch.setitem(sys.modules, "woof", SimpleNamespace(kernel_plan=engine))
    assert kernel_plan.install(plan_dir=tmp_path / "unseeded", tag_text="first-shape")
    assert not kernel_plan._state["plan"].exists()
    return kernel_static, compiler, engine


def test_first_shape_capture_compiles_once_and_foreground_joins(static_setup, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from test_kernel_plan import _wait
    static, compiler, engine = static_setup
    gate = threading.Event()
    compiler.gate = gate
    mutation = []
    def factory():
        try:
            compiler._compile_using_nvrtc_no_warning("slow", ("--fmad=false",), "89", "physics.cu")
        except Exception:
            mutation.append("loader-fallback")
        mutation.append("published")
    monkeypatch.setattr(static, "_selection", lambda run: [("physics", factory, ())])
    discovery = static.start(object())
    discovery.result(timeout=5)
    assert mutation == []
    assert static.receipt()[0]["submitted"] == 1
    assert not kernel_plan._requested
    assert not kernel_plan._state["plan"].exists()
    pool = ThreadPoolExecutor(1)
    requested = pool.submit(compiler._compile_using_nvrtc_no_warning,
                            "slow", ("--fmad=false",), "89", "physics.cu")
    try:
        assert not requested.done()
        gate.set()
        assert requested.result(timeout=5)[0] == b"image:slow:('--fmad=false',):89"
    finally:
        gate.set()
        pool.shutdown(wait=True)
    assert len(compiler.calls) == 1
    assert kernel_plan.receipt()["calls"]["served"] == 1
    assert not getattr(engine._capture, "on", False)
    assert not getattr(kernel_plan._capture, "on", False)


def test_engine_cache_hit_guard_does_not_publish_or_compile(static_setup, monkeypatch):
    static, compiler, engine = static_setup
    mutations = []
    def factory():
        if getattr(engine._capture, "on", False):
            raise engine._Captured()
        mutations.append("physics-parameters")
    monkeypatch.setattr(static, "_selection", lambda run: [("physics", factory, ())])
    future = static.start(object())
    future.result(timeout=5)
    assert static.start(object()) is future
    record, = static.receipt()
    assert record["cached"] == 1 and record["submitted"] == 0
    assert not compiler.calls and not mutations and not kernel_plan._flight


@pytest.mark.parametrize("mode", ["jitify", "method", "missing-engine-guard", "factory-error"])
def test_failed_capture_preserves_ordinary_foreground(static_setup, monkeypatch, mode):
    static, compiler, engine = static_setup
    if mode == "missing-engine-guard":
        engine._capture = None
    def factory():
        if mode == "factory-error":
            raise ValueError("factory refused")
        compiler._compile_using_nvrtc_no_warning("selected", jitify=mode == "jitify",
                                                method="ptx" if mode == "method" else None)
    monkeypatch.setattr(static, "_selection", lambda run: [("physics", factory, ())])
    with pytest.warns(RuntimeWarning, match="foreground compiler remains active"):
        static.start(object()).result(timeout=5)
    record, = static.receipt()
    assert len(record["errors"]) == 1
    assert record["submitted"] == 0 and not compiler.calls
    compiler._compile_using_nvrtc_no_warning("selected")
    assert len(compiler.calls) == 1


def test_other_compiler_owner_does_not_submit(static_setup, monkeypatch):
    static, compiler, _ = static_setup
    kernel_plan._state["installed"] = False
    monkeypatch.setattr(static, "_selection", lambda run: pytest.fail("must not discover"))
    assert static.start(object()) is None
    assert not compiler.calls


def test_already_requested_image_is_not_compiled_again(static_setup, monkeypatch):
    static, compiler, _ = static_setup
    result = compiler._compile_using_nvrtc_no_warning("selected")
    def factory():
        compiler._compile_using_nvrtc_no_warning("selected")
    monkeypatch.setattr(static, "_selection", lambda run: [("physics", factory, ())])
    static.start(object()).result(timeout=5)
    record, = static.receipt()
    assert record["known"] == 1 and record["submitted"] == 0
    assert compiler._compile_using_nvrtc_no_warning("selected") == result
    assert len(compiler.calls) == 1


def test_discovery_is_background_work(static_setup, monkeypatch):
    static, compiler, _ = static_setup
    entered, release = threading.Event(), threading.Event()
    def select(run):
        entered.set()
        assert release.wait(5)
        return []
    monkeypatch.setattr(static, "_selection", select)
    future = static.start(object())
    try:
        assert entered.wait(5)
        assert not future.done()
        assert not compiler.calls
    finally:
        release.set()
        future.result(timeout=5)


def test_compiler_failure_is_receipted_and_foreground_retries(static_setup, monkeypatch):
    from test_kernel_plan import _wait
    static, compiler, _ = static_setup
    compiler.fail_on = "bad"
    def factory():
        compiler._compile_using_nvrtc_no_warning("bad")
    monkeypatch.setattr(static, "_selection", lambda run: [("bad", factory, ())])
    with pytest.warns(RuntimeWarning, match="foreground compiler remains active"):
        static.start(object()).result(timeout=5)
        _wait(lambda: static.receipt()[0]["pool_failed"] == 1 and len(static.receipt()[0]["errors"]) == 1)
    compiler.fail_on = None
    assert compiler._compile_using_nvrtc_no_warning("bad")[0] == b"image:bad:():89"
    assert kernel_plan.receipt()["calls"]["failed_then_inline"] == 1


def test_tag_binds_float_and_nested_configuration(monkeypatch):
    from dataclasses import dataclass, field
    from woof.hex import kernel_static
    @dataclass(frozen=True)
    class Config:
        p_top: float
        options: dict = field(default_factory=dict)
    base = kernel_static._tag(Config(5000.0, {"scheme": 1}))
    assert base != kernel_static._tag(Config(5500.0, {"scheme": 1}))
    assert base != kernel_static._tag(Config(5000.0, {"scheme": 2}))


def test_exact_warm_artifact_avoids_process_creation(static_setup, monkeypatch):
    from woof.hex import kernel_warm
    static, compiler, _ = static_setup
    compiler.__name__ = "cupy.cuda.compiler"
    expected = (b"exact-image", {"f": "mangled"})
    def factory():
        compiler._compile_using_nvrtc_no_warning("warm", (), "89", "warm.cu")
    monkeypatch.setattr(static, "_selection", lambda run: [("warm", factory, ())])
    monkeypatch.setattr(kernel_warm, "restore", lambda arguments, arch: expected)
    monkeypatch.setattr(kernel_plan, "_process_compiler_identity", lambda arch: pytest.fail("unused process identity"))
    static.start(object()).result(timeout=5)
    record, = static.receipt()
    assert record["restored"] == 1 and record["submitted"] == 0
    assert compiler._compile_using_nvrtc_no_warning("warm", (), "89", "warm.cu") == expected
    assert compiler.calls == []


def test_real_process_submission_keeps_exact_request_contract(static_setup, monkeypatch):
    from woof.hex import kernel_warm, kernel_compile_worker
    from woof.hex.cuda_backend import aot
    from test_kernel_plan import _wait
    static, compiler, _ = static_setup
    submissions = []
    identity = {"compiler": "bound"}
    monkeypatch.setattr(kernel_plan, "_process_compiler_identity", lambda arch: identity)
    monkeypatch.setattr(kernel_warm, "worker_config", lambda: {"root": "owned-images", "tag": "bound-tag"})
    monkeypatch.setattr(aot, "effective_options", lambda options: tuple(options))
    class ProcessPool:
        def __init__(self, workers, mp_context, initializer, initargs):
            assert workers == 2 and mp_context.get_start_method() == "spawn"
            # an RTX 5090 host, 2026-10-29: the static pool's workers end with this process too
            assert initializer is kernel_compile_worker.bind_to_parent and initargs == (os.getpid(),)
        def submit(self, function, arguments, arch, compiler_identity, warm):
            assert function is kernel_compile_worker.compile_request
            assert threading.current_thread().name == "kernel-compile-spawner"
            submissions.append((arguments, arch, compiler_identity, warm))
            pending = Future()
            pending.set_result(((b"process-image", None), dict(pid=123, seconds=0.1, cpu_seconds=0.08)))
            return pending
        def shutdown(self, wait, **_kwargs):
            assert wait is False
    monkeypatch.setattr(static.concurrent.futures, "ProcessPoolExecutor", ProcessPool)
    def factory():
        compiler._compile_using_nvrtc_no_warning("process", ("--ftz=false",), "89", "named.cu",
                                                cache_in_memory=True)
    monkeypatch.setattr(static, "_selection", lambda run: [("process", factory, ())])
    static.start(object()).result(timeout=5)
    record, = static.receipt()
    arguments, arch, measured, warm = submissions[0]
    assert arguments["source"] == "process" and arguments["options"] == ["--ftz=false"]
    assert arguments["filename"] == "named.cu" and arguments["cache_in_memory"] is True
    assert arch == "89" and measured == identity and warm["tag"] == "bound-tag"
    assert record["compiler_executor"] == "process" and record["pool_compiled"] == 1
    assert compiler._compile_using_nvrtc_no_warning("process", ("--ftz=false",), "89", "named.cu",
                                                   cache_in_memory=True) == (b"process-image", None)
    assert not compiler.calls


def test_selection_filters_wrf_dynamics_and_keeps_selected_inline_helpers(monkeypatch):
    import sys
    from dataclasses import dataclass
    from woof.hex import kernel_static
    @dataclass
    class Run:
        p_top: float = 5000.0
    factory = lambda: None
    engine_prewarm = SimpleNamespace(
        single_domain_experiment=lambda run: SimpleNamespace(domains=[SimpleNamespace(run=run)]),
        static_jobs=lambda exp: [(label, factory, ()) for label in
            ("acoustic", "wsm6", "unselected", "rrtmgp_rte[RRTMGP_MAX_LAYERS=64]",
             "rrtmgp_validation", "woof.core.tendency_coupling")])
    preflight = SimpleNamespace(CORE_KERNEL_MODULES={"acoustic"},
        physics_kernel_modules=lambda exp: {"acoustic", "wsm6", "rrtmgp_rte"})
    monkeypatch.setitem(sys.modules, "woof", SimpleNamespace(kernel_prewarm=engine_prewarm))
    monkeypatch.setitem(sys.modules, "woof.core", SimpleNamespace(preflight=preflight))
    jobs = kernel_static._selection(Run())
    assert [job[0] for job in jobs] == ["wsm6", "rrtmgp_rte[RRTMGP_MAX_LAYERS=64]",
                                      "rrtmgp_validation", "woof.core.tendency_coupling"]
