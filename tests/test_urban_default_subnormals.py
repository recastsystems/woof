"""Default urban arithmetic must retain positive IEEE subnormal inputs."""
import numpy as np
import pytest

from conftest import requires_gpu


def test_source_scoped_policy_normalizes_cache_and_final_options():
    from types import SimpleNamespace
    from woof.cuda_subnormals import KEY_OPTION, MARKER, install
    calls = []
    class Program:
        def __init__(self, source):
            self.src = source
        def compile(self, options=(), log_stream=None):
            calls.append(('final', options))
    def cache(source, options, *args, **kwargs):
        calls.append(('cache', options))
        # CuPy's preprocessing program has no source marker.
        return Program('').compile(options + ('-ftz=true',))
    compiler = SimpleNamespace(_NVRTCProgram=Program, _compile_with_cache_cuda=cache)
    install(compiler)
    install(compiler)
    request = ('-std=c++17', '-fmad=true', '-ftz=true', '--ftz=false')
    compiler._compile_with_cache_cuda(MARKER + 'urban source', request)
    # The cache key carries KEY_OPTION, so an image an unhooked process wrote
    # for the same source and options (flushed by CuPy's -ftz=true) is never
    # read back by a hooked request.
    assert calls == [('cache', ('-std=c++17', '-fmad=true', KEY_OPTION, '--ftz=false')),
                     ('final', ('-std=c++17', '-fmad=true', KEY_OPTION, '--ftz=false'))]
    calls.clear()
    compiler._compile_with_cache_cuda('other source', ('-std=c++17',))
    assert calls == [('cache', ('-std=c++17',)),
                     ('final', ('-std=c++17', '-ftz=true'))]


def test_install_rehooks_after_another_hook_restores_cupy():
    """A hook that saves and restores CuPy's functions (kernel_aot.capture_requests)
    must not leave marked units compiling with flush-to-zero for the rest of the
    process."""
    from types import SimpleNamespace
    from woof.cuda_subnormals import KEY_OPTION, MARKER, install
    calls = []
    class Program:
        def __init__(self, source):
            self.src = source
        def compile(self, options=(), log_stream=None):
            calls.append(options)
    plain_compile = Program.compile
    def cache(source, options, *args, **kwargs):
        return Program(source).compile(options + ('-ftz=true',))
    compiler = SimpleNamespace(_NVRTCProgram=Program, _compile_with_cache_cuda=cache)
    install(compiler)
    # Another hook restores what it saved before this one was installed.
    compiler._compile_with_cache_cuda = cache
    Program.compile = plain_compile
    install(compiler)
    compiler._compile_with_cache_cuda(MARKER + 'urban source', ('-std=c++17', '--ftz=false'))
    assert calls == [('-std=c++17', KEY_OPTION, '--ftz=false')]
    # Installing again over a live hook does not stack a second one.
    hooked = compiler._compile_with_cache_cuda
    install(compiler)
    assert compiler._compile_with_cache_cuda is hooked


@pytest.mark.parametrize("name", ["urban_ucm", "urban_bep", "urban_bep_couple"])
def test_urban_requests_no_ftz(name):
    from woof.core.kernels import module_options
    assert "--ftz=false" in module_options(name)


@requires_gpu
def test_bem_selects_positive_subnormal_fraction():
    import cupy as cp
    from woof.core.urban_bem import urban_columns
    values = np.array([0, -0., -1.e-40, 1.e-45, 1.e-40, 1.e-37, 1], np.float32)
    got = cp.asnumpy(urban_columns(cp.asarray(values.reshape(1, -1))))
    np.testing.assert_array_equal(got, np.flatnonzero(values > 0))


@requires_gpu
@pytest.mark.parametrize("name", ["urban_ucm", "urban_bep", "urban_bep_couple", "urban_bep_bem"])
def test_final_nvrtc_option_retains_subnormals(name, monkeypatch, tmp_path):
    import cupy as cp
    from cupy.cuda import compiler
    from woof.core.kernels import load_module
    # A private kernel cache: a unit an earlier test compiled into the shared
    # cache never reaches NVRTC, which left nothing to inspect ("run with a
    # fresh CUPY_CACHE_DIR").  CuPy 13 reads CUPY_CACHE_DIR on every compile;
    # CuPy 14 binds it once into a module-level disk backend.
    cache = str(tmp_path / "cupy-cache")
    monkeypatch.setenv("CUPY_CACHE_DIR", cache)
    if hasattr(compiler, "_kernel_cache_backend"):
        monkeypatch.setattr(compiler, "_kernel_cache_backend",
                            compiler._DiskKernelCacheBackend(cache))
    # RawModule compilation is also memoized in-process per device; a unit
    # an earlier test in this process compiled would never reach NVRTC.
    import cupy._util
    cupy._util.clear_memo()
    # A warmed kernel plan also holds images in memory, independently of
    # CuPy's memo and disk cache. This capture must reach NVRTC in a battery
    # that has already compiled these urban units.
    from woof import kernel_plan
    monkeypatch.setattr(kernel_plan, "_flight", {})
    seen = []
    original = cp.cuda.nvrtc.compileProgram

    def capture(program, options):
        seen.append(tuple(options))
        return original(program, options)

    monkeypatch.setattr(cp.cuda.nvrtc, "compileProgram", capture)
    if name == 'urban_bep_bem':
        from woof.core.urban_bem import _bem_module
        _bem_module.cache_clear()
        _bem_module().compile()
    else:
        load_module.cache_clear()
        load_module(name).compile()
    assert seen, "run with a fresh CUPY_CACHE_DIR"
    for options in seen:
        ftz = [o for o in options if o.lstrip('-').split('=')[0] == 'ftz']
        assert ftz == ["--ftz=false"], options
