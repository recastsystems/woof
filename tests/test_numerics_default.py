"""Fast numerics is the default; strict is the opt-in (owner ruling 2026-10-16).  CPU only."""
# Replaces: test_strict_and_wrf_exact_compile_as_written (now test_strict_preserves_options_and_wrf_exact_normalizes_before_lookup).
import importlib
import subprocess
import sys
import types


def _fresh(monkeypatch, **env):
    for name in ("GPUWM_NUMERICS", "GPUWM_WRF_EXACT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    import woof.numerics as numerics
    numerics._mode = None
    return numerics


def test_default_is_fast_and_drops_only_the_no_contraction_request(monkeypatch):
    numerics = _fresh(monkeypatch)
    assert numerics.mode() == "fast"
    assert numerics.effective_options(("-std=c++17", "--fmad=false", "--ftz=false")) == \
        ("-std=c++17", "--ftz=false")
    assert numerics.effective_options(("-fmad=false",)) == ()
    numerics._mode = None


def test_strict_preserves_options_and_wrf_exact_normalizes_before_lookup(monkeypatch):
    # The default (CUDA) backend: strict passes a declared option tuple
    # through unchanged, and WRF-exact normalizes before its lookup.
    tuple_ = ("-std=c++17", "--fmad=false", "--ftz=false")
    numerics = _fresh(monkeypatch, GPUWM_NUMERICS="strict")
    assert numerics.effective_options(tuple_) == tuple_
    numerics = _fresh(monkeypatch, GPUWM_WRF_EXACT="1")
    from woof import wrf_exact
    assert numerics.mode() == "strict"
    assert numerics.effective_options(tuple_) == wrf_exact.effective_options(tuple_)
    assert numerics.effective_options(numerics.effective_options(tuple_)) == \
        numerics.effective_options(tuple_)
    numerics._mode = None


def test_strict_preserves_unit_controls_and_wrf_exact_normalizes_before_lookup(monkeypatch):
    tuple_ = ("-std=c++17", "--fmad=false", "--ftz=false")
    monkeypatch.setenv("WOOF_BACKEND", "amd")
    numerics = _fresh(monkeypatch, GPUWM_NUMERICS="strict")
    assert numerics.effective_options(tuple_) == tuple_
    other = ("--use_fast_math", "--fmad=true", "-ftz=true", "--prec-div=false")
    assert numerics.effective_options(other) == other
    numerics = _fresh(monkeypatch, GPUWM_WRF_EXACT="1")
    from woof import wrf_exact
    assert numerics.mode() == "strict"
    assert numerics.effective_options(tuple_) == wrf_exact.effective_options(tuple_)
    assert numerics.effective_options(numerics.effective_options(tuple_)) == \
        numerics.effective_options(tuple_)
    numerics._mode = None


def test_compile_boundary_hook(monkeypatch):
    numerics = _fresh(monkeypatch)
    seen = []

    class Program:
        def compile(self, options=(), log_stream=None):
            seen.append(("program", tuple(options)))

    compiler = types.SimpleNamespace(
        _compile_with_cache_cuda=lambda source, options, *a, **k: seen.append(("cache", tuple(options))),
        _NVRTCProgram=Program)
    numerics.install(compiler)
    numerics.install(compiler)  # idempotent
    compiler._compile_with_cache_cuda("src", ("-std=c++17", "--fmad=false"))
    Program().compile(("--fmad=false", "--ftz=false"))
    assert seen == [("cache", ("-std=c++17",)), ("program", ("--ftz=false",))]
    numerics._mode = None


def test_import_gpuwm_does_not_import_cupy():
    code = "import sys, woof; print('cupy' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**__import__("os").environ, "GPUWM_NUMERICS": "fast"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_preparation_identity_keeps_contraction_disabled_in_fast_mode(monkeypatch):
    numerics = _fresh(monkeypatch)
    options = ("-std=c++17", "-fmad=false", "-DGPUWM_PREPARATION_IDENTITY=1")
    assert numerics.mode() == "fast"
    assert numerics.effective_options(options) == options
    assert numerics.effective_options(numerics.effective_options(options)) == options
    numerics._mode = None


def test_strict_preserves_final_compiler_options_on_hip(monkeypatch):
    monkeypatch.setenv("WOOF_BACKEND", "amd")
    numerics = _fresh(monkeypatch, GPUWM_NUMERICS="strict")
    seen=[]
    class Program:
        def compile(self,options=(),log_stream=None):seen.append(tuple(options))
    compiler=types.SimpleNamespace(_compile_with_cache_cuda=lambda *a,**kw:None,
                                  _NVRTCProgram=Program,_gpuwm_hip_backend=True)
    numerics.install(compiler)
    Program().compile(("--fmad=true","--ftz=true","--prec-div=false","--prec-sqrt=false"))
    assert seen==[("--fmad=true","--ftz=true","--prec-div=false","--prec-sqrt=false")]
    numerics._mode=None


def test_hip_rawmodule_matches_cupy_ftz_append_but_direct_compile_does_not(monkeypatch):
    from woof import hip_backend
    seen = []
    class Module:
        def load(self, image):
            assert image == b'image'
    monkeypatch.setitem(sys.modules, 'cupy.cuda', types.SimpleNamespace(
        function=types.SimpleNamespace(Module=Module)))
    monkeypatch.setattr(hip_backend, 'active', lambda: True)
    monkeypatch.setattr(hip_backend, 'initialize_module', lambda module: module)
    def compile_source(source, options=(), *args, **kwargs):
        seen.append(tuple(options))
        return b'image', None
    monkeypatch.setattr(hip_backend, 'compile_source', compile_source)
    monkeypatch.setattr(hip_backend, '_ORIGINAL_DIRECT', None)
    compiler = types.SimpleNamespace(
        _compile_using_nvrtc_no_warning=lambda *args, **kwargs: None,
        _compile_module_with_cache=lambda *args, **kwargs: None)
    hip_backend.install(compiler)
    options = ('-std=c++17', '--fmad=false', '--ftz=false')
    compiler._compile_module_with_cache('src', options)
    compiler._compile_using_nvrtc_no_warning('src', options)
    assert seen == [options + ('-ftz=true',), options]
