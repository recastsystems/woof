"""A160: a direct NVRTC compile carries its options in its source.

THE BREAKAGE THIS PREVENTS.  With a GPU visible, NVRTC 13.4 answers a
compile for sm_120 from NVIDIA's ComputeCache (``~/.nv/ComputeCache``, or
``CUDA_CACHE_PATH``) under a key that does not include ``-ftz``.  Measured
on test host D's RTX 5070 Ti (NVRTC 13.4.92, driver 13.2) with a kernel that
divides every float in [1, 2) by the literal 3, one program name, one
source: compiled
with ``-ftz=true`` in one process and then with ``--ftz=false`` in a fresh
process sharing the cache, the second compile returns the first one's cubin
(2,796,202 quotients one ULP off, the A146 reciprocal multiply), and in the
other order the ``-ftz=true`` compile returns the IEEE division.  With
``CUDA_CACHE_DISABLE=1`` or the GPU hidden each flag is honoured.  A
comment naming the flag does not separate the two entries; a device
constant holding it does.  ``--fmad`` was honoured under a fixed name too
(CuPy's in-memory route, which names every program '').

CuPy's own routes are not exposed, measured the same way both orders:
``RawModule`` names the program after a hash of the source and every
option, and ``compile_using_nvrtc`` names it after a fresh temporary
directory, so neither ever meets another option set's entry
(``tests/test_nvrtc_cache_key.py`` holds both).  What is exposed is a
compile that calls NVRTC itself with a fixed program name: the A146
compiler census and SASS tools.  They compile through
:func:`compile_program`, which appends one line to the source::

    extern "C" __device__ const char gpuwm_nvrtc_options[] = "<options>";

The line is data that no kernel reaches, so no instruction changes; it is
appended, so every source line keeps its number (``-lineinfo`` and the
A146 census read line numbers).  Its text is every option passed, a
superset of every option that changes code, so two compiles that differ in
any option differ in their translation unit and no cache can serve one
for the other.  Retire this module when NVRTC keys its cache on ``-ftz``:
the mechanism test in ``tests/test_nvrtc_cache_key.py`` fails on that day.
"""
from __future__ import annotations

#: The C identifier of the appended constant.
SYMBOL = "gpuwm_nvrtc_options"

#: The byte that separates options inside the constant (never in an option).
SEPARATOR = "\x1f"


def _c_string(text: str) -> str:
    out = []
    for ch in text:
        if ch in '\\"':
            out.append("\\" + ch)
        elif " " <= ch <= "~":
            out.append(ch)
        else:
            # A hex escape swallows every hex digit after it; close the
            # literal so the next character starts a new one.
            out.append('\\x%02x""' % (ord(ch) & 0xFF))
    return "".join(out)


def stamp(options) -> str:
    """The line :func:`keyed_source` appends for ``options``."""
    text = SEPARATOR.join(str(option) for option in tuple(options or ()))
    return ('extern "C" __device__ const char %s[] = "%s";'
            % (SYMBOL, _c_string(text)))


def keyed_source(source: str, options) -> str:
    """``source`` with its options line appended, once."""
    line = stamp(options) + "\n"
    if source.endswith(line):
        return source
    sep = "" if not source or source.endswith("\n") else "\n"
    return source + sep + line


def compile_program(source: str, options, *, name: str = "unit.cu",
                    target: str = "ptx", nvrtc=None) -> bytes:
    """Compile ``source`` with NVRTC under ``options`` and return the PTX
    (``target="ptx"``) or cubin (``target="cubin"``) bytes.  The source is
    :func:`keyed_source`'s, so the ComputeCache cannot answer it with an
    entry compiled under other options.  ``nvrtc`` is the backend module
    (``cupy.cuda.nvrtc`` when omitted): a caller that resolved its own, or
    a test standing one in, compiles through the same keyed route."""
    if target not in ("ptx", "cubin"):
        raise ValueError(f"target must be 'ptx' or 'cubin', not {target!r}")
    from woof.hip_backend import active,compile_source
    if active():
        blob,_=compile_source(keyed_source(source,options),options,filename=name)
        return blob
    if nvrtc is None:
        from cupy.cuda import nvrtc

    options = [str(option) for option in tuple(options or ())]
    program = nvrtc.createProgram(keyed_source(source, options), name, [], [])
    try:
        nvrtc.compileProgram(program, options)
        blob = (nvrtc.getPTX(program) if target == "ptx"
                else nvrtc.getCUBIN(program))
    finally:
        nvrtc.destroyProgram(program)
    return blob
