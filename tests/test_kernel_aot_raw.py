"""Final NVRTC requests must replay with their exact filename, flags and aliases."""
import base64
import hashlib
import sys
from types import ModuleType

import pytest

from woof import kernel_aot as aot, kernel_plan, numerics
from woof.kernel_pack import write


@pytest.fixture
def compiler(monkeypatch):
    compiler = ModuleType("cupy.cuda.compiler")
    compiler.calls = []
    def compile_request(*args, **kwargs):
        compiler.calls.append((args, kwargs))
        return b"fallback", None
    compiler._compile_using_nvrtc_no_warning = compile_request
    compiler._compile_with_cache_cuda = compile_request
    compiler.compile_using_nvrtc = compile_request
    compiler._get_arch = lambda: "120"
    compiler._get_arch_for_options_for_nvrtc = lambda arch: ("-arch=sm_" + arch, "cubin")
    compiler._get_bool_env_variable = lambda *args: True
    cp = ModuleType("cupy")
    cp.__file__ = "/engine/site-packages/cupy/__init__.py"
    cuda = ModuleType("cupy.cuda")
    cuda.compiler = compiler
    cp.cuda = cuda
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.compiler", compiler)
    monkeypatch.delenv("WOOF_AOT", raising=False)
    monkeypatch.delenv("WOOF_AOT_RECORD", raising=False)
    monkeypatch.delenv("GPUWM_WRF_EXACT", raising=False)
    monkeypatch.setattr(numerics, "_mode", "strict")
    return compiler


def request():
    return dict(route=aot.RAW_ROUTE, source="template<typename T> __global__ void f(T*) {}",
                options=["--std=c++17", *numerics.STRICT_CONTROLS],
                filename="original.cu", name_expressions=["f<float>", "f<double>"])


def selected(tmp_path, monkeypatch):
    entry = aot.recipe(request())
    image = b"\x7fELFexact-image"
    entry.update(arch="sm_120", cubin_base64=base64.b64encode(image).decode(),
                 cubin_sha256=hashlib.sha256(image).hexdigest(),
                 name_mapping={"f<float>": "symbol_float", "f<double>": "symbol_double"})
    path = tmp_path / "pack.json"
    write(path, {"test": 1}, [entry])
    pack = aot.Pack(path, identity={"test": 1})
    monkeypatch.setattr(aot, "Pack", lambda: pack)
    return pack, image


def test_raw_builder_keeps_final_flags_filename_and_aliases(compiler, monkeypatch):
    from woof import kernel_aot_build, kernel_compile_worker
    from woof.certify import compile_platform
    captured = []
    def compile_request(arguments, target, identity):
        captured.append((arguments, target, identity))
        return ((b"\x7fELFimage", {"f<float>": "symbol"}), {"seconds": 0.01})
    monkeypatch.setattr(kernel_compile_worker, "compile_request", compile_request)
    monkeypatch.setattr(kernel_aot_build, "compiler_identity", lambda: {"test": 1})
    monkeypatch.setattr(compile_platform, "nvrtc_banner", lambda: "test")
    output = kernel_aot_build.compile_entry(aot.recipe(request()), "sm_120")
    arguments, target, identity = captured[0]
    assert arguments == {key: value for key, value in request().items() if key != "route"} | {
        "options": tuple(request()["options"]), "arch": "120", "cache_in_memory": False, "method": None}
    assert target == "120" and identity == {"test": 1}
    assert output["name_mapping"] == {"f<float>": "symbol"}


def test_builder_binds_header_bytes_and_declines_a_changed_install(compiler, monkeypatch, tmp_path):
    from woof import kernel_aot_build, kernel_compile_worker
    from woof.certify import compile_platform
    headers = tmp_path / 'include'
    headers.mkdir()
    header = headers / 'owned.h'
    header.write_text('#define VALUE 1\n')
    monkeypatch.setattr(kernel_compile_worker, 'compile_request',
        lambda *a: ((b'\x7fELFimage', None), {'seconds': 0.01}))
    monkeypatch.setattr(kernel_aot_build, 'compiler_identity', lambda: {'test': 1})
    monkeypatch.setattr(kernel_aot_build, '_HEADER_SNAPSHOTS', {})
    monkeypatch.setattr(compile_platform, 'nvrtc_banner', lambda: 'test')
    args = request() | {'options': [*request()['options'], '-I' + str(headers)]}
    entry = kernel_aot_build.compile_entry(aot.recipe(args), 'sm_120')
    target = tmp_path / 'pack.json'
    write(target, {'test': 1}, [entry])
    key = aot.recipe(args)['module_key']
    assert aot.Pack(target, identity={'test': 1}).select(
        key, args['source'], args['options'], 'sm_120', metadata=entry['compile_request'])[1] is None
    header.write_text('#define VALUE 2\n')
    assert aot.Pack(target, identity={'test': 1}).select(
        key, args['source'], args['options'], 'sm_120', metadata=entry['compile_request'])[1] == 'header_tree_mismatch'


@pytest.mark.parametrize("route", ["worker", "builder"])
def test_cpu_compile_preserves_final_strict_options_in_fast_process(compiler, monkeypatch, route):
    from woof import kernel_aot_build, kernel_compile_worker
    from woof.certify import compile_platform

    monkeypatch.setattr(numerics, "_mode", "fast")
    monkeypatch.setattr(compiler, "_use_ptx", False, raising=False)
    backend = ModuleType("cupy_backends.cuda.libs")
    backend.nvrtc = ModuleType("nvrtc")
    backend.nvrtc.getSupportedArchs = lambda: [120]
    monkeypatch.setitem(sys.modules, "cupy_backends.cuda.libs", backend)
    monkeypatch.setattr(kernel_compile_worker, "_identity", lambda: {"test": 1})
    monkeypatch.setattr(kernel_aot_build, "compiler_identity", lambda: {"test": 1})
    monkeypatch.setattr(compile_platform, "nvrtc_banner", lambda: "test")
    selector = compiler._get_arch_for_options_for_nvrtc
    observed = []

    def program_compile(source, options, target, filename, names, *unused):
        # The engine's real Program.compile hook runs below raw request capture.
        observed.append(numerics.effective_options(options))
        assert compiler._get_arch_for_options_for_nvrtc(target) == ("-arch=sm_120", "cubin")
        if source == "error":
            raise RuntimeError("compile failed")
        return b"\x7fELFimage", {"f<float>": "symbol"}

    monkeypatch.setattr(compiler, "_compile_using_nvrtc_no_warning", program_compile)

    def run(row):
        if route == "builder":
            return kernel_aot_build.compile_entry(aot.recipe(row), "sm_120")
        arguments = {key: value for key, value in row.items() if key != "route"}
        arguments.update(cache_in_memory=False, method=None)
        return kernel_compile_worker.compile_request(arguments, "120", {"test": 1})

    strict = request()
    fast = strict | {"options": ["--std=c++17", "--ftz=false", "--fmad=true"]}
    for row in (strict, fast):
        run(row)
        assert observed[-1] == tuple(row["options"])
        assert numerics._mode == "fast"
        assert compiler._get_arch_for_options_for_nvrtc is selector
    with pytest.raises(RuntimeError, match="compile failed"):
        run(strict | {"source": "error"})
    assert observed[-1] == tuple(strict["options"])
    assert numerics._mode == "fast"
    assert compiler._get_arch_for_options_for_nvrtc is selector


def test_raw_pack_exact_match_and_safe_misses(compiler, tmp_path, monkeypatch):
    pack, image = selected(tmp_path, monkeypatch)
    aot._raw_bridge(sys.modules["cupy"], pack)
    raw = compiler._compile_using_nvrtc_no_warning
    args = {key: value for key, value in request().items() if key != "route"}
    result = raw(**args)
    assert result == (image, {"f<float>": "symbol_float", "f<double>": "symbol_double"})
    result[1].clear()
    assert raw(**args)[1] and not compiler.calls
    for changed in (dict(filename="other.cu"), dict(options=["--std=c++14", *numerics.STRICT_CONTROLS]),
                    dict(name_expressions=list(reversed(args["name_expressions"]))),
                    dict(arch="89"), dict(method="ptx")):
        assert raw(**(args | changed)) == (b"fallback", None)
    monkeypatch.setattr(numerics, "_mode", "fast")
    assert raw(**args) == (b"fallback", None)  # strict image cannot serve fast final flags
    monkeypatch.setattr(numerics, "_mode", "strict")
    compiler._get_arch_for_options_for_nvrtc = lambda arch: ("-arch=compute_" + arch, "ptx")
    assert raw(**args) == (b"fallback", None)


def test_static_capture_precedes_raw_pack_lookup_and_recording(compiler, tmp_path, monkeypatch):
    pack, image = selected(tmp_path, monkeypatch)
    monkeypatch.setenv("WOOF_AOT_RECORD", str(tmp_path / "catalog.json"))
    monkeypatch.setattr(aot, "_RECORDED", {})
    callbacks = []
    import atexit
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(kernel_plan, "_compiler", lambda: compiler)
    monkeypatch.setattr(kernel_plan, "compiler_identity", lambda: {"test": 1})
    monkeypatch.setattr(kernel_plan, "_device_arch", lambda: "120")
    monkeypatch.setattr(kernel_plan, "_state", {"installed": False, "counts": {}})
    monkeypatch.setattr(kernel_plan, "_requested", {})
    monkeypatch.setattr(kernel_plan, "_flight", {})
    monkeypatch.setattr(kernel_plan, "_warm_images", lambda: None)
    monkeypatch.setattr("woof.kernel_warm_pack.install_metadata", lambda *args: None)
    assert kernel_plan.install(tmp_path / "plans", "raw-test")
    args = {key: value for key, value in request().items() if key != "route"}
    with monkeypatch.context() as capture:
        capture.setattr(kernel_plan._capture, "on", True, raising=False)
        with pytest.raises(kernel_plan._Captured) as captured:
            compiler._compile_using_nvrtc_no_warning(**args)
    assert captured.value.arguments["filename"] == "original.cu"
    assert not compiler.calls and not aot._RECORDED and not kernel_plan._requested
    kernel_plan._state.update(compiler_pool=object(), replay={})
    def unexpected_compile(*unused):
        raise AssertionError('shipped raw image bypassed into process compilation')
    monkeypatch.setattr(kernel_plan, '_submit_and_wait', unexpected_compile)
    assert compiler._compile_using_nvrtc_no_warning(**args)[0] == image
    assert not compiler.calls
    raw = [entry for entry in aot._RECORDED.values() if entry.get("route") == aot.RAW_ROUTE]
    assert raw == [request() | {"cache_in_memory": False, "method": None}]
    aot._RECORDED.clear()
    assert compiler._compile_using_nvrtc_no_warning(**args)[0] == image
    assert len(aot._RECORDED) == 1  # a single-flight hit must still appear in the catalog
    held = kernel_plan._aot_coverage()
    arguments = captured.value.arguments
    assert kernel_plan._aot_cached(arguments, "120", held)
    assert not kernel_plan._aot_cached(arguments | {"filename": "different.cu"}, "120", held)
    kernel_plan._state.update(aot_coverage=held, replay={})
    kernel_plan._flight.clear()
    aot._RECORDED.clear()
    assert kernel_plan.ahead(compiler._compile_using_nvrtc_no_warning, **args) == "cached"
    assert kernel_plan._state["replay"]["static_skipped_aot"] == 1
    assert not compiler.calls and not aot._RECORDED and not kernel_plan._flight


@pytest.mark.parametrize("schedule", ["recorded_plan", "static_ahead"])
@pytest.mark.parametrize("pack_case", ["valid", "metadata_mismatch", "corrupt_image"])
def test_plan_aot_hint_uses_real_pack_validation_and_single_flight(
        compiler, tmp_path, monkeypatch, pack_case, schedule):
    """A scheduling hint cannot bypass the real pack's final validation."""
    import json
    from woof import kernel_pack, kernel_warm_pack

    identity = {"test": 1}
    entry = aot.recipe(request())
    image = b"\x7fELFexact-image"
    mapping = {"f<float>": "symbol_float", "f<double>": "symbol_double"}
    entry.update(arch="sm_120", cubin_base64=base64.b64encode(image).decode(),
                 cubin_sha256=hashlib.sha256(image).hexdigest(), name_mapping=mapping)
    if pack_case == "metadata_mismatch":
        # Coverage deliberately uses a subset of the full request metadata.
        entry["compile_request"]["additional_contract"] = "must-match"
    elif pack_case == "corrupt_image":
        # The envelope is valid; the embedded image fails its own digest.
        entry["cubin_sha256"] = "0" * 64
    path = tmp_path / "pack.json"
    write(path, identity, [entry])
    monkeypatch.setenv("WOOF_AOT_PACK", str(path))
    monkeypatch.setattr(aot, "compiler_identity", lambda: identity)
    monkeypatch.setattr(aot, "_PACKS", {})
    reads, selections, images, marks, restores = [], [], [], [], []
    real_read, real_select, real_image = kernel_pack.read, aot.Pack.select, aot.Pack.image

    def read(*args, **kwargs):
        result = real_read(*args, **kwargs)
        reads.append(result)
        return result

    def select(self, *args, **kwargs):
        result = real_select(self, *args, **kwargs)
        selections.append(result[1])
        return result

    def image_bytes(row):
        result = real_image(row)
        images.append(result)
        return result

    class EmptyImages:
        def get(self, arguments):
            restores.append(arguments["source"])
            return None
        def put(self, arguments, result):
            pass
        def close(self):
            pass

    monkeypatch.setattr(kernel_pack, "read", read)
    monkeypatch.setattr(aot.Pack, "select", select)
    monkeypatch.setattr(aot.Pack, "image", staticmethod(image_bytes))
    monkeypatch.setattr(aot, "mark", lambda *args, **kwargs: marks.append((args, kwargs)))
    monkeypatch.setattr(kernel_plan, "_compiler", lambda: compiler)
    monkeypatch.setattr(kernel_plan, "compiler_identity", lambda: identity)
    monkeypatch.setattr(kernel_plan, "_current_device", lambda: None)
    monkeypatch.setattr(kernel_plan, "_device_arch", lambda: "120")
    monkeypatch.setattr(kernel_plan, "_cache_root", lambda: tmp_path / "cupy-cache")
    monkeypatch.setattr(kernel_plan, "_process_compiler_identity", lambda arch: None)
    monkeypatch.setattr(kernel_plan, "_warm_images", lambda: EmptyImages())
    monkeypatch.setattr(kernel_warm_pack, "install_metadata", lambda *args: None)
    monkeypatch.setattr(kernel_plan.atexit, "register", lambda function: None)
    monkeypatch.setattr(kernel_plan, "_state", {"installed": False, "counts": {},
                                               "replay": None, "device_arch": None})
    monkeypatch.setattr(kernel_plan, "_requested", {})
    monkeypatch.setattr(kernel_plan, "_flight", {})
    monkeypatch.delenv(kernel_plan.ENVIRONMENT, raising=False)
    assert kernel_plan.install(tmp_path / "plans", "real-pack-validation")
    assert getattr(kernel_plan._state["original"], "_gpuwm_aot_raw_bridge", False)
    assert reads == [[entry]]  # Actual envelope and compiler identity validation ran.
    args = {key: value for key, value in request().items() if key != "route"}
    arguments = kernel_plan._arguments(**args, arch=None, cache_in_memory=False, method=None)
    plan = kernel_plan._state["plan"]
    if schedule == "recorded_plan":
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps({"schema": kernel_plan.SCHEMA, "compiler": identity,
            "entries": [{"sha256": kernel_plan._key(arguments), "arguments": arguments}]}))
    replay = kernel_plan.start()
    try:
        if schedule == "static_ahead":
            assert replay["plan_source"] == "none" and replay["entries"] == 0
            assert kernel_plan.submit(kernel_plan.ahead,
                lambda: compiler._compile_using_nvrtc_no_warning(**args)).result(5) == "cached"
            assert replay["static_skipped_aot"] == 1
        else:
            assert replay["plan_source"] == "tag"
        assert replay["pool_skipped_aot"] == int(schedule == "recorded_plan")
        assert replay["pool_compiled"] == 0
        assert not restores and not compiler.calls and not selections
        expected = (image, mapping) if pack_case == "valid" else (b"fallback", None)
        raw = compiler._compile_using_nvrtc_no_warning
        assert raw(**args) == expected
        assert raw(**args) == expected
        assert len(compiler.calls) == int(pack_case != "valid")
        assert len(kernel_plan._flight) == 1
        assert kernel_plan._state["counts"]["served"] == 1
        assert selections == (["compile_request_mismatch"] if pack_case == "metadata_mismatch" else [None])
        assert images == ([] if pack_case == "metadata_mismatch" else [image if pack_case == "valid" else None])
        assert len(marks) == 1
        expected_reason = {"metadata_mismatch": "compile_request_mismatch",
                           "corrupt_image": "cubin_digest_mismatch"}.get(pack_case)
        assert marks[0][0][0] == ("aot" if pack_case == "valid" else "jit")
        assert marks[0][1].get("reason") == expected_reason
        if compiler.calls:
            actual, kwargs = compiler.calls[0]
            assert actual[:5] == (args["source"], tuple(args["options"]), None,
                                  args["filename"], args["name_expressions"])
            assert actual[5:] == (None, False, None, None) and not kwargs
        print(json.dumps({"case": pack_case, "schedule": schedule, "pack_reads": len(reads),
            "static_aot_skips": replay.get("static_skipped_aot", 0),
            "aot_plan_skips": replay["pool_skipped_aot"], "plan_image_restores": 0,
            "full_selections": len(selections), "image_checks": len(images),
            "fallback_calls": len(compiler.calls), "calls_served_from_flight": 1,
            "rejection_reason": expected_reason}, sort_keys=True))
    finally:
        kernel_plan.close(wait=True)
