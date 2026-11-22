"""Retain subnormals for marked translation units, including CuPy's final options.

CuPy adds -ftz=true after RawModule options. Normalize both the cache key
and the NVRTC invocation, scoped by a source marker so other units keep
their existing arithmetic. Strict verification can wrap this hook too.
"""
from __future__ import annotations

import functools
from contextvars import ContextVar

MARKER = "#define GPUWM_RETAIN_SUBNORMALS 1\n"
#: Carried only by hooked requests, so their CuPy cache key differs from an
#: unhooked request for the same source and options. Without it the two keys
#: were equal (the units already ask for --ftz=false; CuPy appends -ftz=true
#: only at compile time), and a flush-to-zero image written by any process
#: that compiled a marked unit without this hook was read back by hooked ones
#: (3.0.0 pass 4: the shared gate cache turned the urban wrf_471 water balance
#: arms red). The define has no other effect on the unit.
KEY_OPTION = "-DGPUWM_RETAIN_SUBNORMALS_KEY=1"
_ACTIVE = ContextVar("gpuwm_retain_subnormals_compile", default=False)


def effective_options(source, options):
    if MARKER not in source and not _ACTIVE.get():
        return tuple(options)
    return tuple(o for o in options
                 if o.lstrip("-").split("=", 1)[0] != "ftz" and o != KEY_OPTION) + (KEY_OPTION, "--ftz=false")


def _hooked(function) -> bool:
    """Whether ``function`` is, or wraps, this module's hook."""
    seen = set()
    while function is not None and id(function) not in seen:
        if getattr(function, "_gpuwm_retain_subnormals_hook", False):
            return True
        seen.add(id(function))
        function = getattr(function, "__wrapped__", None)
    return False


def install(compiler=None):
    """Hook CuPy's cache and final NVRTC call, once per live function.

    The check reads the live functions, not a flag on the module: another
    hook that saves and later restores ``_compile_with_cache_cuda`` (for
    example ``woof.kernel_aot.capture_requests``) drops this one, and a
    flag would then keep it off for the rest of the process, and a marked
    urban unit compiled after that would flush its subnormals.
    """
    if compiler is None:
        from cupy.cuda import compiler
    cache_hooked = _hooked(compiler._compile_with_cache_cuda)
    program_hooked = _hooked(compiler._NVRTCProgram.compile)
    if cache_hooked and program_hooked:
        compiler._gpuwm_retain_subnormals = True
        return
    original_cache = compiler._compile_with_cache_cuda
    original_program = compiler._NVRTCProgram.compile

    @functools.wraps(original_cache)
    def cached(source, options, *args, **kwargs):
        # CuPy preprocesses an empty program before compiling this source.
        # It carries the same options but no marker, so retain the policy
        # for all NVRTC calls in this cache request, isolated per context.
        token = _ACTIVE.set(MARKER in source)
        try:
            return original_cache(source, effective_options(source, options),
                                  *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    @functools.wraps(original_program)
    def program(self, options=(), log_stream=None):
        return original_program(self, effective_options(self.src, options), log_stream)

    cached._gpuwm_retain_subnormals_hook = True
    program._gpuwm_retain_subnormals_hook = True
    if not cache_hooked:
        compiler._compile_with_cache_cuda = cached
    if not program_hooked:
        compiler._NVRTCProgram.compile = program
    compiler._gpuwm_retain_subnormals = True
