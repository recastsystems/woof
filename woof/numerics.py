"""Kernel numerics: ``fast`` (the default) or ``strict`` (the opt-in for word-for-word WRF identity).

Owner ruling 2026-10-16: strict numerics is opt-in, never the default.

Many units pin ``--fmad=false`` (and some ``--ftz=false``) in their literal compile tuples because their
column oracles grade them word for word against WRF.  Those tuples stay as written (the oracles, the kernel
manifest audit and ``GPUWM_WRF_EXACT=1`` read them); in ``fast`` mode the compile boundary drops the
no-contraction request, so NVRTC contracts multiply-adds (its own default).  Subnormal requests
(``--ftz=false``) are kept as each unit asks: the land and urban units name a physics breakage without them
(woof/core/ieee_module.py: drainage, snow branches, QSFC; urban rain clamps).  ``--use_fast_math`` is never
added, and IEEE division and square root stay NVRTC's default.

Dedicated prepared-input modules marked ``-DGPUWM_PREPARATION_IDENTITY=1`` keep
their no-contraction request in either mode: changing the preparation backend
must not change the child's SINT terrain capture. Forecast modules remain fast.

``strict`` compiles every tuple exactly as written.  ``GPUWM_WRF_EXACT=1`` implies strict (its own installer
then pins every unit to WRF's arithmetic).  The mode is read once per process from ``GPUWM_NUMERICS``.

The boundary is applied before every compiled-image cache key is formed (CuPy's disk cache and the direct
NVRTC cache, woof/nvrtc_ptx_cache.py), so fast and strict images never share a cache entry.
"""
from __future__ import annotations

import functools
import os
import sys

ENV = "GPUWM_NUMERICS"
MODES = ("fast", "strict")
DEFAULT = "fast"

_NO_CONTRACTION = ("--fmad=false", "-fmad=false", "--fmad=False", "-fmad=False")


def _resolve() -> str:
    if os.environ.get("GPUWM_WRF_EXACT", "0") == "1":
        return "strict"
    value = (os.environ.get(ENV) or DEFAULT).strip().lower()
    if value not in MODES:
        raise ValueError(f"{ENV} must be fast or strict, not {value!r}")
    return value


_mode: str | None = None


def mode() -> str:
    """The process's mode, read from the environment at the first compile and fixed from then on (one
    process runs one mode: a caller such as woof hex's --numerics sets ``GPUWM_NUMERICS`` before it)."""
    global _mode
    if _mode is None:
        _mode = _resolve()
    return _mode


def select(requested: str) -> str:
    """Select at launch, or verify a mode already fixed by kernel compilation."""
    if requested not in MODES:
        raise ValueError(f"numerics must be fast or strict, not {requested!r}")
    if _mode is None:
        if os.environ.get("GPUWM_WRF_EXACT", "0") == "1" and requested != "strict":
            raise ValueError("GPUWM_WRF_EXACT=1 requires strict numerics")
        os.environ[ENV] = requested
    actual = mode()
    if actual != requested:
        raise ValueError(f"process numerics is already {actual}; requested {requested} needs a fresh process")
    return actual


def is_strict() -> bool:
    return mode() == "strict"


def effective_options(options) -> tuple[str, ...]:
    """Final arithmetic options, including opt-in WRF controls, before cache lookup."""
    options = tuple(options)
    # Prepared inputs have one arithmetic contract across CPU and CUDA.
    # A dedicated preparation module may pin that contract without changing
    # the forecast's fast mode. Keep the marker in both compiler cache keys.
    preparation_identity = "-DGPUWM_PREPARATION_IDENTITY=1" in options
    if mode() != "strict" and not preparation_identity:
        options = tuple(option for option in options if option not in _NO_CONTRACTION)
    if os.environ.get("GPUWM_WRF_EXACT", "0") == "1":
        # The program hook applies these even to direct NVRTC. Its image
        # caches and warm-plan/pack identities must see the same controls.
        from woof.wrf_exact import effective_options as wrf_options
        options = wrf_options(options)
    return options


def receipt() -> dict:
    return {"mode": mode(), "env": ENV,
            "fast_drops": list(_NO_CONTRACTION) if mode() == "fast" else [],
            "compiler_hook_installed": bool(_STATE.get("installed"))}


_STATE: dict = {}


def install(compiler=None) -> None:
    """Apply :func:`effective_options` at CuPy's cached compile and at NVRTC's program compile.

    Idempotent; a pass-through in strict mode.  ``compiler`` is ``cupy.cuda.compiler``."""
    if compiler is None:
        from cupy.cuda import compiler
    if not (hasattr(compiler, "_compile_with_cache_cuda") and hasattr(compiler, "_NVRTCProgram")):
        return
    if getattr(compiler, "_gpuwm_numerics_fast", False):
        _STATE["installed"] = True
        return
    original_cache = compiler._compile_with_cache_cuda
    original_program = compiler._NVRTCProgram.compile

    @functools.wraps(original_cache)
    def cached(source, options, *args, **kwargs):
        return original_cache(source, effective_options(options), *args, **kwargs)

    @functools.wraps(original_program)
    def program(self, options=(), log_stream=None):
        return original_program(self, effective_options(options), log_stream)

    compiler._compile_with_cache_cuda = cached
    compiler._NVRTCProgram.compile = program
    compiler._gpuwm_numerics_fast = True
    _STATE["installed"] = True


class _CompilerImportHook:
    """Install the fast boundary the moment ``cupy.cuda.compiler`` is imported, so a default process that
    never imports CuPy does not import it here (``import woof`` stays light)."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "cupy.cuda.compiler":
            return None
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        loader = spec.loader

        class _Loader:
            def create_module(self, spec):
                return loader.create_module(spec)

            def exec_module(self, module):
                loader.exec_module(module)
                try:
                    install(module)
                except Exception as error:  # the strict route still runs; say so loudly
                    print(f"woof: fast numerics could not be installed ({type(error).__name__}: {error}); "
                          "this process compiles strict", file=sys.stderr, flush=True)

        spec.loader = _Loader()
        return spec


def arm() -> None:
    """Called by ``import woof``: install now if CuPy's compiler is loaded, else when it is.  Nothing is
    armed under ``GPUWM_WRF_EXACT=1`` (its own installer owns the compile boundary)."""
    if os.environ.get("GPUWM_WRF_EXACT", "0") == "1":
        return
    module = sys.modules.get("cupy.cuda.compiler")
    if module is not None:
        install(module)
        return
    if not any(isinstance(finder, _CompilerImportHook) for finder in sys.meta_path):
        sys.meta_path.insert(0, _CompilerImportHook())
