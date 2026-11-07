"""Real driver loads and real JIT fallbacks, with byte-level result checks."""
import json

import pytest

pytestmark = pytest.mark.gpu
SOURCE = 'extern "C" __global__ void copy_words(const unsigned* a, unsigned* b) { b[threadIdx.x] = a[threadIdx.x]; }'


def test_real_module_load_and_named_jit_fallbacks(tmp_path, monkeypatch):
    import cupy as cp
    import numpy as np
    write = pytest.importorskip("woof.hex.kernel_pack").write
    from woof.hex.cuda_backend import KernelCache, require_cuda
    from woof.hex.cuda_backend.aot import compiler_identity, source_hash
    from woof.hex.cuda_backend.aot_build import compile_entry
    from cupy.cuda import compiler

    capability = require_cuda()
    row = {"module_key": "hexcore.aot_test", "source": SOURCE,
           "source_sha256": source_hash(SOURCE),
           "options": ["--std=c++17", "--fmad=false"]}
    packed = compile_entry(row, capability.sm)
    path = tmp_path / "pack.json"
    write(path, compiler_identity(), [packed])
    monkeypatch.setenv("WOOF_HEX_AOT_PACK", str(path))
    monkeypatch.delenv("WOOF_HEX_AOT", raising=False)
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("WOOF_TIME_LEDGER", str(ledger))
    monkeypatch.setenv("WOOF_PRODUCT", "hex")
    words = np.array([0, 1, 0x80000001, 0x7f800000, 0x7fc00001], dtype=np.uint32)
    original = compiler._compile_using_nvrtc_no_warning
    compiled = []
    def count(source, *args, **kwargs):
        if source == SOURCE or source.startswith(SOURCE):
            compiled.append(source)
        return original(source, *args, **kwargs)
    monkeypatch.setattr(compiler, "_compile_using_nvrtc_no_warning", count)
    results = []
    for index, (source, architectures) in enumerate([
        (SOURCE, [packed]),
        (SOURCE + "\n// changed source", [packed]),
        (SOURCE, [{**packed, "arch": "sm_100" if capability.sm == "sm_120" else "sm_120"}]),
    ]):
        write(path, compiler_identity(), architectures)
        cache = KernelCache(capability=capability, cache_dir=tmp_path / str(index))
        kernel = cache.raw_kernel("copy_words", source, module_key=row["module_key"])
        output = cp.empty_like(cp.asarray(words))
        kernel((1,), (len(words),), (cp.asarray(words), output))
        cp.cuda.runtime.deviceSynchronize()
        results.append(output.get().tobytes())
        if index == 0:
            assert compiled == []
    assert len(compiled) == 2
    assert results == [words.tobytes()] * 3
    marks = [json.loads(line)["attrs"] for line in ledger.read_text().splitlines()]
    # The process-wide CuPy hook also emits loads; this checks the named cache's exact three decisions.
    loads = [m for m in marks if m.get("mark") == "kernel.load" and m.get("module_key") == row["module_key"]]
    assert [(m["source"], m.get("reason")) for m in loads] == [
        ("aot", None), ("jit", "source_hash_mismatch"), ("jit", "architecture_mismatch")]


def test_real_cupy_bridge_preserves_template_name_mapping(tmp_path, monkeypatch):
    import cupy as cp
    import numpy as np
    from cupy.cuda import compiler
    write = pytest.importorskip("woof.hex.kernel_pack").write
    from woof.hex.cuda_backend.aot import Pack, capture_requests, compiler_identity, install, recipe
    from woof.hex.cuda_backend.aot_build import compile_entry

    source = 'typedef unsigned int word; template<typename T> __global__ void aot_copy_template(const T* a, T* b) { b[threadIdx.x] = a[threadIdx.x]; }'
    names = ["aot_copy_template<unsigned int>"]
    options = ("--std=c++17", "--fmad=false")
    with capture_requests() as requests:
        cp.RawModule(code=source, options=options, name_expressions=names).get_function(names[0])
        cp.RawModule(code=source, options=options, name_expressions=["aot_copy_template<word>"]).get_function("aot_copy_template<word>")
    captured = list(requests.values())
    assert len(captured) == 2
    cp.clear_memo()
    row = recipe(captured[0])
    entry = compile_entry(row, "sm_" + cp.cuda.Device().compute_capability)
    alias_names = ["aot_copy_template<word>"]
    alias = compile_entry(recipe(captured[1]),
                          "sm_" + cp.cuda.Device().compute_capability)
    path = tmp_path / "template-pack.json"
    write(path, compiler_identity(), [entry, alias])
    original = compiler._compile_with_cache_cuda
    # An earlier GPU test in this worker (any KernelCache over an installed pack that carries
    # CuPy compile requests) leaves the process-wide bridge here, and install() keeps the first
    # bridge it finds, so this test's pack would never be read (seal3 hex GPU suite, -n 8).
    while getattr(original, "_hex_aot_bridge", False):
        original = original.__wrapped__
    link = original
    while link is not None:
        assert not getattr(link, "_hex_aot_bridge", False), "an AOT bridge sits under another wrapper"
        link = getattr(link, "__wrapped__", None)
    monkeypatch.setattr(compiler, "_compile_with_cache_cuda", original)
    install(cp, Pack(path))
    calls = []
    compile_original = compiler._compile_using_nvrtc_no_warning
    def count(text, *a, **kw):
        if text.startswith(source):
            calls.append(text)
        return compile_original(text, *a, **kw)
    monkeypatch.setattr(compiler, "_compile_using_nvrtc_no_warning", count)
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "empty-cupy"))
    if hasattr(compiler, "_set_kernel_cache_backend"):
        # CuPy 14 read CUPY_CACHE_DIR at import; the variable alone does not empty its disk cache.
        from cupy.cuda._compiler_cache import DiskKernelCacheBackend
        monkeypatch.setattr(compiler, "_kernel_cache_backend", DiskKernelCacheBackend(str(tmp_path / "empty-cupy")))
    results = []
    words = np.array([1, 0x80000001, 0x7fc00001], dtype=np.uint32)
    first = cp.RawModule(code=source, options=options, name_expressions=names)
    first.get_function(names[0])
    second = cp.RawModule(code=source, options=options, name_expressions=alias_names)
    second.get_function(alias_names[0])
    first.get_function(names[0])
    assert calls == []
    for text in (source, source + "\n// changed template source"):
        module = cp.RawModule(code=text, options=options, name_expressions=names)
        kernel = module.get_function(names[0])
        out = cp.empty(words.shape, dtype=cp.uint32)
        kernel((1,), (len(words),), (cp.asarray(words), out))
        results.append(out.get().tobytes())
        if text == source:
            assert calls == []
    assert calls == [source + "\n// changed template source"]
    assert results == [words.tobytes()] * 2


def test_real_cached_banner_preserves_fingerprint_without_compiling(tmp_path, monkeypatch):
    from cupy_backends.cuda.libs import nvrtc
    from woof.certify import compile_platform
    write = pytest.importorskip("woof.hex.kernel_pack").write
    from woof.hex.cuda_backend.aot import Pack, cache_compiler_banner, compiler_identity, recipe
    from woof.hex.cuda_backend.aot_build import compile_entry
    import cupy as cp

    original_banner = compile_platform.nvrtc_banner
    monkeypatch.setattr(compile_platform, "nvrtc_banner", original_banner)
    fingerprint = compile_platform.compile_platform_fingerprint()
    banner = compile_platform.nvrtc_banner()
    entry = compile_entry(recipe({"source": SOURCE, "options": ["--std=c++17", "--fmad=false"]}),
                          "sm_" + cp.cuda.Device().compute_capability)
    entry["nvrtc_banner"] = banner
    path = tmp_path / "banner-pack.json"
    write(path, compiler_identity(), [entry])
    calls = []
    original_compile = nvrtc.compileProgram
    def count(*args, **kwargs):
        calls.append(True)
        return original_compile(*args, **kwargs)
    monkeypatch.setattr(nvrtc, "compileProgram", count)
    cache_compiler_banner(Pack(path))
    assert compile_platform.compile_platform_fingerprint() == fingerprint
    assert calls == []
