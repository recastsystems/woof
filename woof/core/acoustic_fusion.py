"""One launch per acoustic phase group: the substep's kernels joined by text.

The split-explicit substep used to submit up to six kernels per acoustic
iteration (apply_emdiv, advance_uv, advance_mu_th[_msf], advance_w_phi[_msf],
the specified-frame update and the sumflux accumulation).  Each re-read the
column it had just written.  This module appends two entries to the
UNCHANGED acoustic translation unit:

``acoustic_uv_substep``
    apply_emdiv's body, then advance_uv's body.  Both kernels use the same
    union-grid thread mapping (one thread per (k, j, i) of the u and v
    faces) and read and write only their own u''/v'' element, so the
    external-mode increment and the momentum update stay in one thread.

``acoustic_column_substep`` (selectable; off by default because it measured
slower than the separate column kernels, see ``JOIN_COLUMN`` below)
    advance_mu_th[_msf], advance_w_phi[_msf], the specified-frame phi update,
    the frame's zero-gradient w copy and the sumflux accumulation, one mass
    column per thread.  Each phase reads only its own column of anything an
    earlier phase of the same launch wrote.  The one exception in the
    separate kernels -- the frame thread copying the just-solved w'' of its
    interior source column -- is turned around: the interior source column
    writes the copy into the frame columns it feeds.  The copy moves words,
    it computes nothing.

How the arithmetic is kept.  Every phase is the original kernel body text
(the preprocessor arm the default build compiles), in its own scope, with a
phase-local ``return`` turned into ``break``.  An empty ``asm volatile`` with
a memory clobber separates the phases: every value that used to cross a
kernel boundary still crosses through memory, so no stored word is forwarded
into a later phase's expression and no multiply-add contraction can join two
phases.  Every global store of the separate kernels is kept.  The fused
module is compiled with the acoustic module's own options.

This is the default arithmetic build only.  The strict build, the
upper-wind limiter, radiative open boundaries (whose u''/v'' face fix-ups sit
between the two kernels) and a specified frame wider than the mass/w frame
keep the separate kernels.  Exact words are a device proof, not a property of
the text: ``tools/acoustic_fusion_replay.py`` runs every substep of a forecast
through both paths and compares every output word.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import re

import numpy as np

from woof.core.device_cache import cuda_cache

#: Set False to submit the separate kernels (the replay harness's reference).
ENABLED = True
#: Which phases the column launch joins.  Every combination writes the same
#: words (tests/test_acoustic_fusion.py); the defaults are the measured
#: fastest.  Per acoustic substep at 300x300x49 (random consistent state
#: with map factors, terrain, moisture, a specified frame, emdiv and
#: sumflux), best of four interleaved runs:
#:
#:   variant (u/v joined in all but the first)   RTX 4090, idle   RTX 5090, shared
#:   separate kernels                             1.632 ms         1.313 ms
#:   column separate                              1.588            1.303
#:   column mu+w                                  1.622            1.320
#:   column mu+w+frame                            1.625            1.291
#:   column mu+w+sumflux                          1.659            1.303
#:   column mu+w+frame+sumflux                    1.669            1.336
#:
#: On the idle 4090 every column join is slower than the separate column
#: kernels (the long serial column threads lose the overlap the separate
#: launches get, and the L2 already serves the re-read column), so the
#: column stays separate by default and the joined column is a knob.
JOIN_COLUMN = False          # advance_mu_th[_msf] + advance_w_phi[_msf]
JOIN_COLUMN_FRAME = True     # + the specified frame and its w copy
JOIN_COLUMN_SUMFLUX = False  # + zero/accumulate_sumflux

UV_ENTRY = "acoustic_uv_substep"
COLUMN_ENTRY = "acoustic_column_substep"
_BARRIER = '\n    asm volatile("" ::: "memory");\n'


class FusionUnsupported(ValueError):
    """The original kernel text no longer has the audited shape."""


def _helpers():
    from woof.ensemble.batch_kernel import _active_source, _close, _masked
    return _active_source, _close, _masked


@lru_cache(maxsize=None)
def acoustic_source() -> str:
    """The acoustic unit as the loader hands it to NVRTC, read once.

    woof.core.kernels.load_module compiles ``acoustic`` once per process
    and device, so the separate kernels a process runs come from this one
    reading too.
    """
    from woof.core.kernels import module_source
    return module_source("acoustic")


@lru_cache(maxsize=None)
def _audit_options():
    """The options the acoustic module compiles with, plus CuPy's arch flag."""
    from woof.hip_backend import active
    if active(): return ("-std=c++17",)
    from cupy.cuda import compiler
    helper = getattr(compiler, "_get_arch_for_options_for_nvrtc", None)
    if helper is None:
        raise FusionUnsupported("installed CuPy does not expose its NVRTC architecture flag")
    flag, _ = helper()
    return ("-std=c++17", flag)


def _entry(active: str, name: str):
    """(launch bounds text, [(declaration, name)], body) of one entry."""
    _, close, masked_of = _helpers()
    masked = masked_of(active)
    bounds = r"(__launch_bounds__\s*\(\s*[1-9][0-9]*\s*(?:,\s*[1-9][0-9]*\s*)?\)\s*)?"
    found = list(re.finditer(r"\b__global__\s+" + bounds + r"void\s+" + name + r"\s*\(",
                             masked))
    if len(found) != 1:
        raise FusionUnsupported(f"{name}: one unconditional __global__ definition is required")
    match = found[0]
    start = match.end() - 1
    end = close(masked, start, "(", ")")
    body_start = masked.index("{", end)
    if masked[end + 1:body_start].strip():
        raise FusionUnsupported(f"{name}: decorated definition")
    body_end = close(masked, body_start, "{", "}")
    params = []
    for part in masked[start + 1:end].split(","):     # comments blanked
        declaration = " ".join(part.split())
        ident = re.search(r"([A-Za-z_]\w*)\s*$", declaration)
        if ident is None or any(token in declaration for token in "[]()&="):
            raise FusionUnsupported(f"{name}: unsupported parameter {declaration!r}")
        params.append((declaration, ident[1]))
    launch_bounds = " ".join(match[1].split()) if match[1] else None
    return launch_bounds, params, active[body_start + 1:body_end]


def _rename(text: str, renames: dict) -> str:
    for old, new in renames.items():
        text = re.sub(r"\b" + old + r"\b", new, text)
    return text


def _phase(name: str, body: str, renames: dict) -> str:
    """One original body as a phase: its own scope, a local return breaks."""
    _, close, masked_of = _helpers()
    masked = masked_of(body)
    if re.search(r"^[ \t]*#", masked, re.M):
        raise FusionUnsupported(f"{name}: a preprocessor line inside the body would leave its scope")
    if re.search(r"\bdo\b", masked):
        raise FusionUnsupported(f"{name}: a do loop's return would leave only the loop")
    returns = list(re.finditer(r"\breturn\s*;", masked))
    for loop in re.finditer(r"\b(?:for|while|switch)\s*\(", masked):
        end_condition = close(masked, loop.end() - 1, "(", ")")
        statement = end_condition + 1
        while masked[statement].isspace():
            statement += 1
        end_statement = (close(masked, statement, "{", "}") if masked[statement] == "{"
                         else masked.index(";", statement))
        if any(statement <= item.start() <= end_statement for item in returns):
            raise FusionUnsupported(f"{name}: a return inside a loop would leave only the loop")
    # A break in an original body is inside its own loop or switch (C has no
    # other kind), so it still leaves only that statement inside the phase.
    for item in reversed(returns):
        body = body[:item.start()] + "break;" + body[item.end():]
    body = _rename(body, renames)
    # The inactive preprocessor arms were blanked in place; drop those lines.
    body = "\n".join(line for line in body.split("\n") if line.strip())
    return f"\n    // ---- phase: {name} (original body) ----\n    do {{\n{body}\n    }} while (false);\n"


def _merge(parameters: list, declarations: dict, params, renames: dict):
    for declaration, name in params:
        if name in renames:
            declaration = re.sub(r"\b" + name + r"\s*$", renames[name], declaration)
            name = renames[name]
        if name not in declarations:
            parameters.append(name)
            declarations[name] = declaration
            continue
        old, new = declarations[name], declaration
        if old.replace("const ", "") != new.replace("const ", ""):
            raise FusionUnsupported(f"parameter {name} is declared {old!r} and {new!r}")
        # A pointer one phase writes stays writable in the joined entry.
        if "*" in new and "const" not in new.split():
            declarations[name] = new


def _written(parameters, declarations) -> list:
    """Names of the joined entry's writable pointer parameters."""
    return [name for name in parameters
            if "*" in declarations[name] and "const" not in declarations[name].split()]


# The specified frame's zero-gradient w copy, as the frame kernel writes it.
_FRAME_W_COPY = (r"w_pp\s*\[\s*tid\s*\]\s*=\s*w_pp\s*\[\s*\(\s*size_t\s*\)\s*k\s*\*\s*st"
                 r"\s*\+\s*source_c\s*\]\s*;")

_W_COPY_PHASE = r"""
    // ---- phase: specified-frame zero-gradient w, written by its source ----
    // advance_specified_phi_w copied w''(k, clamp(j), clamp(i)) into each
    // frame column after advance_w_phi had finished every interior column.
    // In one launch the interior source column writes those same words.
    do {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= ny * nx) break;
        int j = c / nx, i = c - j * nx;
        int z = frame_spec_zone;
        if (i < z || i >= nx - z || j < z || j >= ny - z) break;
        size_t st = (size_t)ny * nx;
        int j0 = (j == z) ? 0 : j, j1 = (j == ny - 1 - z) ? ny - 1 : j;
        int i0 = (i == z) ? 0 : i, i1 = (i == nx - 1 - z) ? nx - 1 : i;
        for (int jj = j0; jj <= j1; ++jj) {
            for (int ii = i0; ii <= i1; ++ii) {
                if (jj >= z && jj < ny - z && ii >= z && ii < nx - z) continue;
                size_t f = (size_t)jj * nx + ii;
                for (int k = 0; k <= nz; ++k)
                    w_pp[(size_t)k * st + f] = w_pp[(size_t)k * st + c];
            }
        }
    } while (false);
"""

_SUMFLUX_PHASE = r"""
    // ---- phase: sumflux accumulation (accumulate_sumflux's element op) ----
    // ru_m += u'', rv_m += v'', ww_m += Omega'' for this column, plus the
    // last u face of a row and the last v row.  zero_sumflux's stage reset
    // is the first substep's +0.0f addend.
    do {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= ny * nx) break;
        int j = c / nx, i = c - j * nx;
        int nxf = nx + 1;
        size_t st = (size_t)ny * nx;
        for (int k = 0; k < nz; ++k) {
            size_t u = ((size_t)k * ny + j) * nxf + i;
            ru_m[u] = arn_add(sumflux_first ? 0.0f : ru_m[u], u_pp[u]);
            if (i == nx - 1)
                ru_m[u + 1] = arn_add(sumflux_first ? 0.0f : ru_m[u + 1], u_pp[u + 1]);
            size_t v = ((size_t)k * (ny + 1) + j) * nx + i;
            rv_m[v] = arn_add(sumflux_first ? 0.0f : rv_m[v], v_pp[v]);
            if (j == ny - 1)
                rv_m[v + nx] = arn_add(sumflux_first ? 0.0f : rv_m[v + nx], v_pp[v + nx]);
        }
        for (int k = 0; k <= nz; ++k) {
            size_t w = (size_t)k * st + c;
            ww_m[w] = arn_add(sumflux_first ? 0.0f : ww_m[w], ww_pp[w]);
        }
    } while (false);
"""

_SUMFLUX_PARAMS = (("real* __restrict__ ru_m", "ru_m"), ("real* __restrict__ rv_m", "rv_m"),
                   ("real* __restrict__ ww_m", "ww_m"), ("int sumflux_first", "sumflux_first"))


def _column_entry(active: str, mapped: bool, frame: bool, sumflux: bool):
    """The joined column entry's text, parameter names and receipt."""
    mu_name = "advance_mu_th_msf" if mapped else "advance_mu_th"
    w_name = "advance_w_phi_msf" if mapped else "advance_w_phi"
    parts = [(mu_name, _entry(active, mu_name), {}),
             (w_name, _entry(active, w_name), {})]
    if frame:
        bounds, params, body = _entry(active, "advance_specified_phi_w")
        _, _, masked_of = _helpers()
        copies = list(re.finditer(_FRAME_W_COPY, masked_of(body)))
        if len(copies) != 1:
            raise FusionUnsupported("advance_specified_phi_w: the frame w copy changed shape")
        copy = copies[0]
        body = body[:copy.start()] + " " * (copy.end() - copy.start()) + body[copy.end():]
        parts.append(("advance_specified_phi_w", (bounds, params, body),
                      {"spec_zone": "frame_spec_zone"}))
    bounds = {part[1][0] for part in parts[:2]}
    launch_bounds = bounds.pop() if len(bounds) == 1 else None
    parameters, declarations, phases = [], {}, []
    for name, (_, params, body), renames in parts:
        _merge(parameters, declarations, params, renames)
        phases.append(_phase(name, body, renames))
    if frame:
        phases.append(_W_COPY_PHASE)
    if sumflux:
        _merge(parameters, declarations, _SUMFLUX_PARAMS, {})
        phases.append(_SUMFLUX_PHASE)
    text = ('\n\n// Joined acoustic column substep (woof/core/acoustic_fusion.py).\n'
            'extern "C" __global__\n' + (launch_bounds + "\n" if launch_bounds else "")
            + "void " + COLUMN_ENTRY + "("
            + ",\n        ".join(declarations[name] for name in parameters) + ")\n{"
            + _BARRIER.join(phases) + "}\n")
    receipt = {"entries": [part[0] for part in parts], "written": _written(parameters, declarations),
               "frame_w_copy": "by_source_column" if frame else None,
               "sumflux": sumflux, "launch_bounds": launch_bounds,
               "original_body_sha256": {part[0]: sha256(part[1][2].encode()).hexdigest() for part in parts}}
    return text, tuple(parameters), receipt


def _uv_entry(active: str):
    """The joined u/v entry's text, parameter names and receipt."""
    parts = [("apply_emdiv", _entry(active, "apply_emdiv"), {"spec_zone": "emdiv_spec_zone"}),
             ("advance_uv", _entry(active, "advance_uv"), {})]
    parameters, declarations, phases = [], {}, []
    for name, (_, params, body), renames in parts:
        _merge(parameters, declarations, params, renames)
        phases.append(_phase(name, body, renames))
    bounds = {part[1][0] for part in parts}
    launch_bounds = bounds.pop() if len(bounds) == 1 else None
    text = ('\n\n// Joined acoustic u/v substep (woof/core/acoustic_fusion.py).\n'
            'extern "C" __global__\n' + (launch_bounds + "\n" if launch_bounds else "")
            + "void " + UV_ENTRY + "("
            + ",\n        ".join(declarations[name] for name in parameters) + ")\n{"
            + _BARRIER.join(phases) + "}\n")
    receipt = {"entries": [part[0] for part in parts], "written": _written(parameters, declarations),
               "launch_bounds": launch_bounds,
               "original_body_sha256": {part[0]: sha256(part[1][2].encode()).hexdigest() for part in parts}}
    return text, tuple(parameters), receipt


_OFF_OPEN = chr(10) + "#if 0  /* separate kernel; the joined entries follow */" + chr(10)
_OFF_CLOSE = chr(10) + "#endif" + chr(10)


def _without_entries(source: str, active: str) -> str:
    """The unit with every original ``__global__`` definition compiled out.

    Each definition (``extern "C" __global__`` through its closing brace) is
    wrapped in ``#if 0``/``#endif`` at its own offsets, read from the
    option-resolved view where its signature arms balance.  Helpers, macros
    and constants keep their text.  Every helper the joined entries call then
    has exactly the call sites it has in the separate kernels' unit (one
    joined entry per original user), so the compiler's inlining choices --
    and with them any multiply-add contraction across a call -- match.
    """
    _, close, masked_of = _helpers()
    masked = masked_of(active)
    edits = []
    for match in re.finditer(r"\bextern\s+__global__\b", masked):
        name = re.compile(r"\bvoid\s+([A-Za-z_]\w*)\s*\(").search(masked, match.end())
        if name is None:
            raise FusionUnsupported("a __global__ definition without a void entry")
        end = close(masked, name.end() - 1, "(", ")")
        body_start = masked.index("{", end)
        body_end = close(masked, body_start, "{", "}")
        original = source[match.start():body_end + 1]
        opened = closed = 0
        for line in masked_of(original).splitlines():
            directive = re.match(r"\s*#\s*(\w+)", line)
            if directive and directive[1] in ("if", "ifdef", "ifndef"):
                opened += 1
            elif directive and directive[1] == "endif":
                closed += 1
                if closed > opened:
                    raise FusionUnsupported(f"{name[1]}: its definition closes an outer conditional")
        if opened != closed:
            raise FusionUnsupported(f"{name[1]}: its definition leaves a conditional open")
        edits.append((match.start(), body_end + 1))
    if not edits:
        raise FusionUnsupported("the acoustic unit has no __global__ definitions")
    for start, end in reversed(edits):
        source = (source[:start] + _OFF_OPEN + source[start:end] + _OFF_CLOSE
                  + source[end:])
    return source


@lru_cache(maxsize=None)
def joined_unit(source: str, mapped: bool, frame: bool, sumflux: bool, options: tuple):
    """One translation unit holding both joined entries for a configuration.

    ``source`` is the text :func:`woof.core.kernels.module_source` gives for
    ``acoustic`` and ``options`` the options it compiles with (plus CuPy's
    architecture flag, which selects nothing in this unit).  Returns
    (unit, u/v parameter names, column parameter names, receipt).
    """
    active_source, _, _ = _helpers()
    active = active_source(source, options)
    uv_text, uv_names, uv_receipt = _uv_entry(active)
    column_text, column_names, column_receipt = _column_entry(active, mapped, frame, sumflux)
    # A joined parameter must not hide a file-scope function a phase calls.
    helpers = set(re.findall(r"\b__device__\b[^;{}()]*?\b([A-Za-z_]\w*)\s*\(",
                             _helpers()[2](active)))
    hidden = helpers & (set(uv_names) | set(column_names))
    if hidden:
        raise FusionUnsupported(f"joined parameters hide device functions {sorted(hidden)}")
    unit = _without_entries(source, active) + uv_text + column_text
    receipt = {"acoustic_unit_sha256": sha256(source.encode()).hexdigest(),
               "joined_unit_sha256": sha256(unit.encode()).hexdigest(),
               "uv": uv_receipt, "column": column_receipt,
               "options": list(options),
               "separate_entries": "compiled out (#if 0) in the joined unit"}
    return unit, uv_names, column_names, receipt


@cuda_cache(maxsize=None)
def _module(source: str):
    """Compile the joined unit exactly as the acoustic module is compiled.

    woof.core.kernels.load_module's plain route: ``cp.RawModule`` with the
    literal ``("-std=c++17",)`` tuple (CuPy appends its flush flag to both).
    """
    import cupy as cp
    from woof.core.kernels import MODULE_KEY_ROOT, _compile_observed
    from woof.certify.kernel_manifest import record_module
    key = f"{MODULE_KEY_ROOT}:acoustic+joined[{sha256(source.encode()).hexdigest()[:16]}]"
    module = cp.RawModule(code=source, options=("-std=c++17",), name_expressions=None)
    _compile_observed(module, key)
    record_module(key, source=source, options=("-std=c++17",), module=module)
    return module


def kernel(source: str, entry: str):
    return _module(source).get_function(entry)


def bind(names, *named: tuple) -> tuple:
    """Arguments for the joined entry from the separate kernels' own tuples.

    ``named`` holds (parameter names, argument tuple, renames) per original
    kernel: each name receives exactly what that kernel was launched with.
    A name two kernels share must have been given the same array or value.
    """
    values = {}
    for params, args, renames in named:
        if len(params) != len(args):
            raise FusionUnsupported("an original kernel's argument count differs from its signature")
        for name, value in zip(params, args):
            name = renames.get(name, name)
            if name in values and not _same(values[name], value):
                raise FusionUnsupported(f"{name} is bound to two different values")
            values.setdefault(name, value)
    missing = [name for name in names if name not in values]
    if missing:
        raise FusionUnsupported(f"unbound joined parameters {missing}")
    return tuple(values[name] for name in names)


def _same(a, b) -> bool:
    if a is b:
        return True
    if isinstance(a, np.generic) and isinstance(b, np.generic):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    data = getattr(a, "data", None)
    other = getattr(b, "data", None)
    if hasattr(data, "ptr") and hasattr(other, "ptr"):
        return (data.ptr == other.ptr and a.shape == b.shape and a.strides == b.strides
                and a.dtype == b.dtype)
    return False


@lru_cache(maxsize=None)
def writable_parameters(source: str, entry: str, options: tuple) -> tuple:
    """Names of an original entry's writable pointer parameters."""
    active_source, _, _ = _helpers()
    return tuple(name for declaration, name in _entry(active_source(source, options), entry)[1]
                 if "*" in declaration and "const" not in declaration.split())


@lru_cache(maxsize=None)
def original_parameters(source: str, entry: str, options: tuple) -> tuple:
    active_source, _, _ = _helpers()
    return tuple(name for _, name in _entry(active_source(source, options), entry)[1])
