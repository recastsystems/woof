"""Byte-gated layout and stage-fusion candidates, separate from the executor.

These candidates preserve the installed kernels' float expressions. Omega's
temporary divergence parking can become thread-local, and flux divergence can
start its accumulator at the RK zero value instead of reading a zeroed field.
The member-innermost stencil adapter changes the complete field-pointer helper
call graph as well as direct loads. No candidate is a forecast qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from types import MappingProxyType

import numpy as np

from woof.ensemble.batch_kernel import (
    BatchKernelUnsupported, KernelSpec, _close, _masked,
    prepare_batch_source_launch,
)

_TPB = 128
_LAYOUTS = ("outermost", "innermost")
LAYOUT_TRIALS = ("none", "outermost", "innermost")
_FIELD_HELPERS = {
    "xface_cell_open": "q", "xface_cell_per": "q",
    "yface_cell_open": "q", "yface_cell_per": "q",
    "zface_half": "q", "zface_w_centered": "w", "w_lid_velocity": "flux",
}


@dataclass(frozen=True)
class TrialLaunch:
    """A prepared launch with evidence of its changed storage operations."""

    operation: object
    metadata: object

    def __call__(self):
        return self.operation()

    @property
    def binding_receipt(self):
        return getattr(self.operation, "binding_receipt", None)


def _trial(operation, **metadata):
    return TrialLaunch(operation, MappingProxyType(metadata))


def _resources(source, spec, binding_members, launch):
    from woof.ensemble.batch_kernel import _compiled_source
    kernel = _compiled_source(source, spec, binding_members,
                              launch.binding_receipt["audit_options"])[0]
    return dict(kernel.attributes)


def _layout(layout):
    if layout not in _LAYOUTS:
        raise ValueError("layout must be outermost or innermost")
    return layout


def pack_member_innermost(array):
    """Copy CUDA words from (N, ...) to contiguous (..., N) storage."""
    import cupy as cp
    from woof.ensemble.batch_fluxes import _cuda_array
    _cuda_array(array, "array", array.ndim)
    if array.ndim < 2:
        raise ValueError("member layout conversion needs a member and spatial axis")
    return cp.ascontiguousarray(cp.moveaxis(array, 0, -1))


def unpack_member_innermost(array, *, out=None):
    """Copy CUDA words from (..., N) to (N, ...), retaining an optional output."""
    import cupy as cp
    from woof.ensemble.batch_fluxes import _cuda_array
    _cuda_array(array, "array", array.ndim)
    if array.ndim < 2:
        raise ValueError("member layout conversion needs a member and spatial axis")
    view = cp.moveaxis(array, -1, 0)
    if out is None:
        return cp.ascontiguousarray(view)
    _cuda_array(out, "out", array.ndim)
    if out.shape != view.shape:
        raise ValueError("unpack output must have the complete member-outermost shape")
    if (view.flags.c_contiguous and out.flags.c_contiguous
            and int(view.data.ptr) == int(out.data.ptr) and view.nbytes == out.nbytes):
        # A contiguous moveaxis view can already expose the requested words.
        # N=1 is the common case; no copy or arithmetic is needed.
        return out
    # Overlapping conversion would overwrite words before their transposed read.
    from woof.ensemble.batch_fluxes import _separate
    _separate(out, (("packed input", array),))
    cp.copyto(out, view)
    return out


def _rewrite_subscripts(text, pointers, transform):
    """Change only addresses, with balanced brackets and no pointer escape."""
    masked = _masked(text)
    edits = []
    for name in pointers:
        for match in re.finditer(r"\b" + re.escape(name) + r"\s*\[", masked):
            start = masked.index("[", match.start(), match.end())
            end = _close(masked, start, "[", "]")
            edits.append((match.start(), end + 1,
                          transform(name, text[start + 1:end])))
    edits.sort()
    previous = 0
    result = []
    for start, end, replacement in edits:
        if start < previous:
            raise BatchKernelUnsupported("nested member-pointer loads need a separate address audit")
        result.extend((text[previous:start], replacement))
        previous = end
    result.append(text[previous:])
    return "".join(result)


def _inner_address(name, index, members):
    return (f"{name}[static_cast<unsigned long long>({index}) * {members}u "
            "+ __trial_member]")


def _local_divergence_operation(operation, nz):
    """Retain every Omega float operation and replace its scratch addresses."""
    stores = "ww[(k + 1) * ncol + col] = divv;"
    reads = "const T divv = ww[at];"
    loops = ("for (int k = 0; k < nz; ++k) {",
             "for (int k = 1; k < nz; ++k) {")
    if (operation.count(stores) != 1 or operation.count(reads) != 1
            or any(operation.count(loop) != 1 for loop in loops)):
        raise BatchKernelUnsupported(
            "Omega parking/loop structure changed; the temporary level ownership needs a new audit")
    result = operation.replace(stores, "__trial_divv[k] = divv;")
    result = result.replace(reads, "const T divv = __trial_divv[k - 1];")
    for loop in loops:
        result = result.replace(loop, "#pragma unroll\n" + loop.replace("k < nz", f"k < {nz}"))
    return f"T __trial_divv[{nz}];\n" + result


def omega_trial_source(nz, members, *, has_msf=False, layout="outermost",
                       cache_shared=False):
    """Build a closed Omega candidate from the installed factory operation.

    The local array is a proposed byte reduction. Compiler local-memory spills
    can add traffic and must be measured before claiming reduced DRAM bytes.
    Shared coefficient staging is block-local and carries no member reduction.
    """
    from woof.ensemble.batch_fluxes import _raw_source, _checked_operation
    from woof.ensemble.batch_kernel import _members
    _layout(layout)
    members = _members(members)
    if isinstance(nz, bool) or not isinstance(nz, (int, np.integer)) or nz < 1:
        raise ValueError("nz must be a positive integer")
    nz = int(nz)
    source, spec, names = _raw_source("omega", bool(has_msf))
    operation = _checked_operation("omega", bool(has_msf))
    if members == 1:
        return source, spec, names
    replacement = _local_divergence_operation(operation, nz)
    if layout == "innermost":
        replacement = _rewrite_subscripts(
            replacement, ("ru", "rv", "ww"),
            lambda name, index: _inner_address(name, index, members))
        coordinates = (
            "const unsigned long long __trial_linear = "
            "static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n"
            f"const unsigned int __trial_member = __trial_linear % {members}u;\n"
            f"const long long i = __trial_linear / {members}u;\n"
            "if (i >= scalar_size) return;\n")
        old = ("const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n"
               "if (i >= scalar_size) return;\n")
        if source.count(old) != 1:
            raise BatchKernelUnsupported("Omega scalar launch coordinates changed")
        source = source.replace(old, coordinates)
    if cache_shared:
        staging = (
            f"__shared__ float __trial_coefficients[{2 * nz}];\n"
            f"for (unsigned int at = threadIdx.x; at < {nz}u; at += blockDim.x) {{\n"
            "  __trial_coefficients[at] = dnw[at];\n"
            f"  __trial_coefficients[{nz}u + at] = c1h[at];\n"
            "}\n__syncthreads();\n"
            "dnw = __trial_coefficients;\n"
            f"c1h = __trial_coefficients + {nz};\n")
        # All block threads stage before the scalar-size guard. The values are
        # immutable shared coefficients, never data from another member.
        entry = source.index("{", source.index("extern \"C\" __global__"))
        source = source[:entry + 1] + "\n" + staging + source[entry + 1:]
    if source.count(operation) != 1:
        raise BatchKernelUnsupported("Omega operation is no longer one contiguous factory body")
    source = source.replace(operation, replacement)
    spec = KernelSpec("ensemble_layout_trials", spec.entry, spec.pointers, spec.options)
    return source, spec, names


def _outer_view(array, layout):
    if layout == "outermost":
        return array
    return array.reshape((1,) + tuple(array.shape[:-1]))


def prepare_omega_trial(ru, rv, ww, dnw, c1h, *, dx, dy, has_msf=False,
                        msft=None, layout="outermost", cache_shared=False):
    """Bind local-divergence Omega, optionally on resident (..., N) fields."""
    from woof.ensemble.batch_fluxes import (
        _cuda_array, _shared, _separate, _scalar_index_limit, _wrf_reciprocal,
        _logical, prepare_omega_columns,
    )
    _layout(layout)
    for name, array in (("ru", ru), ("rv", rv), ("ww", ww)):
        _cuda_array(array, name, 4)
    if layout == "outermost":
        members, nz, ny, nxp1 = map(int, ru.shape)
        expected = ((members, nz, ny + 1, nxp1 - 1),
                    (members, nz + 1, ny, nxp1 - 1))
    else:
        nz, ny, nxp1, members = map(int, ru.shape)
        expected = ((nz, ny + 1, nxp1 - 1, members),
                    (nz + 1, ny, nxp1 - 1, members))
    nx = nxp1 - 1
    if nx < 1 or tuple(rv.shape) != expected[0] or tuple(ww.shape) != expected[1]:
        raise ValueError("Omega inputs need the complete compatible C-grid/member shapes")
    _shared(dnw, "dnw", (nz,))
    _shared(c1h, "c1h", (nz,))
    has_msf = _logical(has_msf, "has_msf")
    inputs = [("ru", ru), ("rv", rv), ("dnw", dnw), ("c1h", c1h)]
    if has_msf:
        _shared(msft, "msft", (ny, nx))
        inputs.append(("msft", msft))
    _separate(ww, inputs)
    _scalar_index_limit(max(ru.size, rv.size, ww.size) // members)
    if members == 1:
        original = prepare_omega_columns(
            _outer_view(ru, layout), _outer_view(rv, layout), _outer_view(ww, layout),
            dnw, c1h, dx=dx, dy=dy, has_msf=has_msf, msft=msft)
        return _trial(original, candidate="omega", layout=layout, members=1,
                      n1_original_factory=True, removed_logical_global_bytes=0)
    rdx, rdy = _wrf_reciprocal(dx, "dx"), _wrf_reciprocal(dy, "dy")
    source, spec, _ = omega_trial_source(
        nz, members, has_msf=has_msf, layout=layout, cache_shared=cache_shared)
    args = [ru, rv, dnw, c1h]
    strides = {"ru": ru.strides[0], "rv": rv.strides[0], "ww": ww.strides[0],
               "dnw": 0, "c1h": 0}
    if has_msf:
        args.append(msft)
        strides["msft"] = 0
    args += [rdx, rdy, np.int32(nz), np.int32(ny), np.int32(nx), ww, np.int32(ny * nx)]
    inner = layout == "innermost"
    count = ny * nx * (members if inner else 1)
    launch = prepare_batch_source_launch(
        source, spec, 1 if inner else members, ((count + _TPB - 1) // _TPB,),
        (_TPB,), tuple(args), pointer_strides=None if inner else strides)
    return _trial(launch, candidate="omega_local_divergence", layout=layout,
                  members=members, shared_coefficients_cached=bool(cache_shared),
                  n1_original_factory=False, source_sha256=sha256(source.encode()).hexdigest(),
                  kernel_attributes=_resources(source, spec, 1 if inner else members, launch),
                  removed_logical_global_bytes=4 * (2 * nz - 1) * members * ny * nx,
                  byte_limit="address traffic removed; local spills and cache traffic require counters")


def _helper_calls(text, member_pointers):
    """Audit every member pointer use against direct loads and known helpers."""
    masked = _masked(text)
    allowed = set()
    calls = []
    for name in _FIELD_HELPERS:
        for match in re.finditer(r"\b" + name + r"\s*\(", masked):
            start = masked.index("(", match.start(), match.end())
            end = _close(masked, start, "(", ")")
            first = re.match(r"\s*([A-Za-z_]\w*)\s*,", masked[start + 1:end])
            if first is None or first[1] not in member_pointers:
                raise BatchKernelUnsupported(f"{name} field pointer is no longer a direct member argument")
            allowed.add(start + 1 + first.start(1))
            calls.append((end, name))
    for name in member_pointers:
        for match in re.finditer(r"\b" + name + r"\b", masked):
            after = masked[match.end():].lstrip()
            if not after.startswith("[") and match.start() not in allowed:
                raise BatchKernelUnsupported(
                    f"member pointer {name} escapes the direct-index/known-helper call graph")
    return calls


def _append_helper_member(text, member_pointers):
    calls = _helper_calls(text, member_pointers)
    for end, _ in sorted(calls, reverse=True):
        text = text[:end] + ", __trial_member" + text[end:]
    return text


def _helper_definition(source, name):
    masked = _masked(source)
    match = re.search(r"__device__\s+__forceinline__\s+real\s+" + name + r"\s*\(", masked)
    if match is None:
        raise BatchKernelUnsupported(f"member helper {name} definition changed")
    start = masked.index("(", match.start(), match.end())
    end = _close(masked, start, "(", ")")
    body_start = masked.index("{", end)
    body_end = _close(masked, body_start, "{", "}")
    return match.start(), start, end, body_start, body_end


def _stencil_cpp_arms(body):
    """Split the two numerical arms without mistaking nested order clauses.

    Each arm owns its bounds guard and exit stores. Horizontal-order
    specializations add nested #if/#else directives inside those arms;
    splitting every #else formerly refused every fused production launch.
    """
    depth = 0
    outer = []
    for match in re.finditer(r"^[ \t]*#(if|ifdef|ifndef|elif|else|endif)\b[^\n]*", body, re.M):
        kind = match.group(1)
        if kind in ("if", "ifdef", "ifndef"):
            if depth == 0:
                outer.append(match)
            depth += 1
        elif kind == "endif":
            depth -= 1
            if depth < 0:
                raise BatchKernelUnsupported("stencil CPP directives are unbalanced")
            if depth == 0:
                outer.append(match)
        elif depth == 1:
            outer.append(match)
    if (depth or [m.group(1) for m in outer] != ["if", "else", "endif"]
            or not outer[0].group().startswith("#if GPUWM_WRF_EXACT_C_ADVECTION")):
        raise BatchKernelUnsupported("stencil CPP arm topology changed; numerical exits need a new audit")
    parts = []
    end = 0
    for match in outer:
        parts.extend((body[end:match.start()], body[match.start():match.end()]))
        end = match.end()
    parts.append(body[end:])
    return parts


def flux_trial_source(source, spec, members, *, layout="outermost", zero_tendency=False):
    """Transform one complete advection entry and its field-reading helpers.

    Float math text remains unchanged. The fused RK zero changes accumulator
    storage to one thread-local FP32 value and stores on every numerical exit.
    All source changes are addresses, integer coordinates, or that storage.
    """
    from woof.ensemble.batch_kernel import _entry_parts, _members
    _layout(layout)
    members = _members(members)
    if members == 1:
        return source
    if spec.module != "advection" or spec.entry not in (
            "flux_div_scalar", "flux_div_u", "flux_div_v", "flux_div_w"):
        raise BatchKernelUnsupported("the stencil trial only audits the four flux-divergence entries")
    declaration, _, _, body_start, body_end, _, _, _ = _entry_parts(source, spec)
    body = source[body_start + 1:body_end]
    member_pointers = tuple(p.name for p in spec.pointers if p.role == "member")
    _helper_calls(body, member_pointers)
    if re.search(r"\b(?:__syncthreads|__syncwarp|atomic\w*|__shfl\w*)\b", _masked(body)):
        raise BatchKernelUnsupported("member-innermost mapping would change stencil collective membership")
    first_entry = re.search(r'extern\s+"C"\s+__global__\s+void\s+flux_div_scalar', source)
    if first_entry is None:
        raise BatchKernelUnsupported("advection helper prefix no longer precedes its scalar entry")
    prefix = source[:first_entry.start()]
    if spec.entry == "flux_div_w":
        lo, _, _, _, hi = _helper_definition(source, "w_lid_velocity")
        prefix += source[lo:hi + 1] + "\n"
    if layout == "innermost":
        for name, pointer in _FIELD_HELPERS.items():
            if name == "w_lid_velocity" and spec.entry != "flux_div_w":
                continue
            lo, _, end, helper_start, hi = _helper_definition(prefix, name)
            helper = prefix[helper_start + 1:hi]
            _helper_calls(helper, (pointer,))
            helper = _rewrite_subscripts(
                helper, (pointer,), lambda name, index: _inner_address(name, index, members))
            prefix = (prefix[:end] + ", unsigned int __trial_member" + prefix[end:helper_start + 1]
                      + helper + prefix[hi:])
        body = _append_helper_member(body, member_pointers)
        coordinates = re.compile(r"int i = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;")
        if len(coordinates.findall(body)) != 2:
            raise BatchKernelUnsupported("stencil x coordinates changed outside the two audited CPP arms")
        body = coordinates.sub(
            "const unsigned long long __trial_x = "
            "static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;\n"
            f"    const unsigned int __trial_member = __trial_x % {members}u;\n"
            f"    int i = __trial_x / {members}u;", body)
        body = _rewrite_subscripts(
            body, tuple(name for name in member_pointers if name != "tend_out" or not zero_tendency),
            lambda name, index: _inner_address(name, index, members))
    if zero_tendency:
        nys = "(ny + 1)" if spec.entry == "flux_div_v" else "ny"
        nxs = "(nx + 1)" if spec.entry == "flux_div_u" else "nx"
        index = f"(static_cast<unsigned long long>(k) * {nys} + j) * {nxs} + i"
        output = (_inner_address("tend_out", index, members) if layout == "innermost"
                  else f"tend_out[{index}]")
        store = f"{output} = __trial_tendency;"
        arms = _stencil_cpp_arms(body)
        for at in (2, 4):
            arm = arms[at]
            guard = re.search(r"if \(i >= [^\n]+\) return;", _masked(arm))
            if guard is None:
                raise BatchKernelUnsupported("stencil bounds guard changed; off-grid stores cannot be inferred")
            tail = arm[guard.end():]
            tail = _rewrite_subscripts(tail, ("tend_out",), lambda name, index: "__trial_tendency")
            tail = re.sub(r"\breturn\s*;", "{ " + store + " return; }", tail)
            arms[at] = arm[:guard.end()] + "\n    real __trial_tendency = 0.0f;\n" + tail + "\n" + store + "\n"
        body = "".join(arms)
    # _entry_parts starts at __global__, after the original C linkage. Keep
    # the plain entry symbol that RawModule.get_function requests.
    return prefix + 'extern "C" ' + source[declaration:body_start + 1] + body + "}\n"


def prepare_flux_div_trial(field, ru, rv, rw, tend, spacing, fnm, fnp, msf, *,
                           dx, dy, stagger="", open_x=False, open_y=False,
                           has_msf=False, spec=False, layout="outermost", zero_tendency=False,
                           vorder=3, horder=5, theta_offset=False):
    """Bind the closed stencil layout and optional adjacent RK-zero fusion.

    With fusion, omit this field's row from the external RK tendency zero. The
    trial overwrites its prior tendency with exactly the zero-plus-advection
    value. Other slow RHS contributions must remain after this launch.
    ``vorder`` is the field's WRF vertical order (3 or 5), the trailing
    argument of the audited entry, as ``prepare_flux_div`` passes it.
    """
    from woof.core.kernels import module_source
    from woof.ensemble import batch_operators as operators
    from woof.ensemble.batch_fluxes import _cuda_array
    _layout(layout)
    if stagger not in operators._FLUX_SPECS:
        raise ValueError("advection staggering must be '', 'x', 'y' or 'z'")
    if vorder not in (3, 5):
        raise ValueError("advection vorder must be 3 or 5 (the WRF vert_order ladders the kernel carries)")
    for name, array in (("field", field), ("ru", ru), ("rv", rv), ("rw", rw), ("tend", tend)):
        _cuda_array(array, name, 4)
    dims = field.shape if layout == "outermost" else (field.shape[-1],) + field.shape[:-1]
    members, nlev, nys, nxs = map(int, dims)
    nz, ny, nx = nlev - int(stagger == "z"), nys - int(stagger == "y"), nxs - int(stagger == "x")
    if min(nz, ny, nx) < 1:
        raise ValueError("advection staggering leaves no mass-point spatial core")
    shapes = ((members, nz, ny, nx + 1), (members, nz, ny + 1, nx),
              (members, nz + 1, ny, nx), tuple(dims))
    for (name, array), shape in zip((("ru", ru), ("rv", rv), ("rw", rw), ("tend", tend)), shapes):
        expected = shape if layout == "outermost" else shape[1:] + shape[:1]
        if tuple(array.shape) != expected:
            raise ValueError(f"{name} needs the complete member/C-grid shape {expected}")
    kernel_spec = operators._FLUX_SPECS[stagger]
    field_name, spacing_name = kernel_spec.pointers[0].name, kernel_spec.pointers[5].name
    for name, profile in ((spacing_name, spacing), ("fnm", fnm), ("fnp", fnp)):
        operators._shared_profile(profile, name, nz)
    operators._cuda_float32(msf, "msf")
    if msf.shape != (nys, nxs):
        raise ValueError("msf needs the target field's shared two-dimensional staggering")
    operators._separate_output(tend, ((field_name, field), ("ru", ru), ("rv", rv),
                                     ("rw", rw), (spacing_name, spacing), ("fnm", fnm),
                                     ("fnp", fnp), ("msf", msf)))
    if (open_x and nx < operators.FIFTH_ORDER_STENCIL_AXIS
            or open_y and ny < operators.FIFTH_ORDER_STENCIL_AXIS):
        raise ValueError("open advection needs axis >= 7 so boundary degrade bands do not overlap")
    dx, dy = operators._finite(dx, "dx"), operators._finite(dy, "dy")
    if dx <= 0 or dy <= 0:
        raise ValueError("advection spacing must be positive")
    if members == 1:
        original = operators.prepare_flux_div(
            *(_outer_view(value, layout) for value in (field, ru, rv, rw, tend)),
            spacing, fnm, fnp, msf, dx=dx, dy=dy, stagger=stagger,
            open_x=open_x, open_y=open_y, has_msf=has_msf, spec=spec, vorder=vorder,
            horder=horder, theta_offset=theta_offset)
        def scalar():
            if zero_tendency:
                tend.fill(0)
            original()
        return _trial(scalar, candidate="advection", layout=layout, members=1,
                      n1_original_entry=True, zero_tendency=bool(zero_tendency),
                      removed_logical_global_bytes=0)
    from woof.core.kernel_shape import shape_defines, specialize_source
    source = specialize_source("advection", _order_source(
                                   operators.advection_order_defines(
                                       horder, theta_offset=theta_offset, stagger=stagger)),
                               shape_defines("advection", kernel_spec.entry, (nz, ny, nx)))
    source = flux_trial_source(source, kernel_spec, members,
                               layout=layout, zero_tendency=zero_tendency)
    trial_spec = KernelSpec("ensemble_layout_trials", kernel_spec.entry,
                            kernel_spec.pointers, kernel_spec.options)
    args = (field, ru, rv, rw, tend, spacing, fnm, fnp, msf,
            operators._kernel_scalar(1.0 / dx, "inverse dx"),
            operators._kernel_scalar(1.0 / dy, "inverse dy"),
            np.int32(nz), np.int32(ny), np.int32(nx), np.int32(open_x),
            np.int32(open_y), np.int32(has_msf), np.int32(spec), np.int32(vorder))
    inner = layout == "innermost"
    grid = (((nxs * (members if inner else 1)) + _TPB - 1) // _TPB, nys, nlev)
    strides = {name: array.strides[0] for name, array in
               ((field_name, field), ("ru", ru), ("rv", rv), ("rw", rw), ("tend_out", tend))}
    strides.update({spacing_name: 0, "fnm": 0, "fnp": 0, "msf": 0})
    launch = prepare_batch_source_launch(
        source, trial_spec, 1 if inner else members, grid, (_TPB, 1, 1), args,
        pointer_strides=None if inner else strides)
    active_levels = nlev - (1 + int(nz < 2) if stagger == "z" else 0)
    return _trial(launch, candidate="flux_divergence", layout=layout, members=members,
                  staggering=stagger, zero_tendency=bool(zero_tendency),
                  n1_original_entry=False, source_sha256=sha256(source.encode()).hexdigest(),
                  kernel_attributes=_resources(source, trial_spec, 1 if inner else members, launch),
                  removed_logical_global_bytes_upper_bound=(
                      8 * members * active_levels * nys * nxs if zero_tendency else 0),
                  byte_limit="zero write/prior read bound; excluded points retain zero stores; DRAM needs counters")


def advection_family_source(source, members, *, layout="outermost", zero_tendency=True,
                            audit_options=None):
    """Dispatch four unchanged advection float bodies in one family launch.

    Each x-block region owns one staggering. Its device body receives scalar
    virtual coordinates, and its original bounds guard excludes excess y/z
    blocks. No output from one staggering is consumed by another staggering.
    N=1 returns the original translation unit and has no new family entry.
    """
    from woof.ensemble.batch_kernel import (
        _entry_parts, _members, PointerSpec, _active_source, _effective_options)
    from woof.ensemble.batch_operators import _FLUX_SPECS
    _layout(layout)
    members = _members(members)
    if members == 1:
        return source, None
    candidates = tuple(flux_trial_source(source, spec, members, layout=layout,
                                         zero_tendency=zero_tendency)
                       for spec in _FLUX_SPECS.values())
    last_spec = _FLUX_SPECS["z"]
    context = _effective_options(last_spec.options) if audit_options is None else tuple(audit_options)
    prefix_end = _entry_parts(candidates[-1], last_spec, context)[0]
    before_entry = candidates[-1][:prefix_end]
    linkage = re.search(r'extern\s+"C"\s*$', before_entry)
    if linkage is None:
        raise BatchKernelUnsupported("audited stencil entry lost C linkage before family extraction")
    prefix = before_entry[:linkage.start()]
    if re.search(r"\b(?:blockIdx|gridDim|threadIdx|blockDim|__shared__|__syncthreads|"
                 r"__syncwarp|atomic\w*|__shfl\w*)\b",
                 _masked(_active_source(prefix, context))):
        raise BatchKernelUnsupported(
            "family field helpers use implicit grid/thread coordinates or shared collectives; "
            "dispatch would give them another staggering's coordinates or member group")
    functions = []
    for candidate, spec in zip(candidates, _FLUX_SPECS.values()):
        _, start, end, body_start, body_end, _, _, _ = _entry_parts(candidate, spec, context)
        body = candidate[body_start + 1:body_end]
        masked = _masked(body)
        replacements = []
        for match in re.finditer(r"\bblockIdx\s*\.\s*([xyz])\b", masked):
            replacements.append((match.start(), match.end(), "__trial_block_" + match[1]))
        for lo, hi, replacement in reversed(replacements):
            body = body[:lo] + replacement + body[hi:]
        if re.search(r"\b(?:blockIdx|gridDim|__shared__|__syncthreads|__syncwarp|atomic\w*|__shfl\w*)\b", _masked(body)):
            raise BatchKernelUnsupported(
                f"{spec.entry}: family dispatch cannot preserve an implicit grid coordinate or collective")
        functions.append("__device__ __forceinline__ void __trial_" + spec.entry + "("
                         + candidate[start + 1:end]
                         + ", unsigned int __trial_block_x, unsigned int __trial_block_y, "
                         "unsigned int __trial_block_z) {\n" + body + "}\n")
    fields = ("theta", "u", "v", "w", "ru", "rv", "rw", "theta_t", "u_t", "v_t", "w_t")
    shared = ("rdnw", "rdn", "fnm", "fnp", "msft", "msfu", "msfv")
    pointers = tuple(PointerSpec(name, "member") for name in fields)
    pointers += tuple(PointerSpec(name, "shared") for name in shared)
    outputs = frozenset(("theta_t", "u_t", "v_t", "w_t"))
    parameters = [f"{'real' if name in outputs else 'const real'}* __restrict__ {name}"
                  for name in fields + shared]
    # The entries' trailing vorder: scalars and w take WRF's
    # v_sca_adv_order (advect_w keys its vertical order on the scalar
    # order), u and v take v_mom_adv_order.
    parameters += ["real dx_inv", "real dy_inv", "int nz", "int ny", "int nx",
                   "int open_x", "int open_y", "int has_msf", "int boundary_spec",
                   "int vorder_scalar", "int vorder_momentum"]
    multiplier = members if layout == "innermost" else 1
    common = "dx_inv, dy_inv, nz, ny, nx, open_x, open_y, has_msf, boundary_spec, "
    coordinates = ", __trial_family_x, blockIdx.y, blockIdx.z"
    calls = (
        "__trial_flux_div_scalar(theta, ru, rv, rw, theta_t, rdnw, fnm, fnp, msft, "
        + common + "vorder_scalar" + coordinates,
        "__trial_flux_div_u(u, ru, rv, rw, u_t, rdnw, fnm, fnp, msfu, "
        + common + "vorder_momentum" + coordinates,
        "__trial_flux_div_v(v, ru, rv, rw, v_t, rdnw, fnm, fnp, msfv, "
        + common + "vorder_momentum" + coordinates,
        "__trial_flux_div_w(w, ru, rv, rw, w_t, rdn, fnm, fnp, msft, "
        + common + "vorder_scalar" + coordinates,
    )
    dispatch = (
        "extern \"C\" __global__ void flux_div_family(" + ", ".join(parameters) + ") {\n"
        f"const unsigned int __trial_scalar_blocks = "
        f"(static_cast<unsigned long long>(nx) * {multiplier}u + {_TPB - 1}u) / {_TPB}u;\n"
        f"const unsigned int __trial_u_blocks = "
        f"((static_cast<unsigned long long>(nx) + 1u) * {multiplier}u + {_TPB - 1}u) / {_TPB}u;\n"
        "unsigned int __trial_family_x = blockIdx.x;\n"
        "if (__trial_family_x < __trial_scalar_blocks) {\n" + calls[0] + "); return; }\n"
        "__trial_family_x -= __trial_scalar_blocks;\n"
        "if (__trial_family_x < __trial_u_blocks) {\n" + calls[1] + "); return; }\n"
        "__trial_family_x -= __trial_u_blocks;\n"
        "if (__trial_family_x < __trial_scalar_blocks) {\n" + calls[2] + "); return; }\n"
        "__trial_family_x -= __trial_scalar_blocks;\n"
        "if (__trial_family_x < __trial_scalar_blocks) {\n" + calls[3] + "); }\n}\n")
    spec = KernelSpec("ensemble_layout_trials", "flux_div_family", pointers)
    return prefix + "\n".join(functions) + dispatch, spec


def prepare_advection_family_trial(rows, ru, rv, rw, fnm, fnp, *, dx, dy,
                                    open_x=False, open_y=False, has_msf=False,
                                    specified=False, layout="outermost", zero_tendency=True,
                                    vorder_scalar=3, vorder_momentum=3,
                                    horder_scalar=5, horder_momentum=5, theta_offset=False):
    """Bind the four-stagger family with disjoint output and shared-field audits.

    ``vorder_scalar`` (WRF v_sca_adv_order) reaches the scalar and w rows,
    ``vorder_momentum`` (v_mom_adv_order) the u and v rows."""
    from woof.core.kernels import module_source
    from woof.ensemble import batch_operators as operators
    from woof.ensemble.batch_fluxes import _cuda_array, _separate, _scalar_index_limit
    _layout(layout)
    rows = tuple(rows)
    if len(rows) != 4 or tuple(row[-1] for row in rows) != ("", "x", "y", "z"):
        raise ValueError("family dispatch needs scalar/u/v/w rows in original order")
    arrays = tuple(row[0] for row in rows) + (ru, rv, rw) + tuple(row[1] for row in rows)
    for index, array in enumerate(arrays):
        _cuda_array(array, f"family field {index}", 4)
    dims = rows[0][0].shape if layout == "outermost" else (rows[0][0].shape[-1],) + rows[0][0].shape[:-1]
    members, nz, ny, nx = map(int, dims)
    shapes = ((members, nz, ny, nx), (members, nz, ny, nx + 1),
              (members, nz, ny + 1, nx), (members, nz + 1, ny, nx))
    def check(array, expected, name):
        shape = expected if layout == "outermost" else expected[1:] + expected[:1]
        if tuple(array.shape) != shape:
            raise ValueError(f"{name} needs complete family/C-grid shape {shape}")
    for row, expected in zip(rows, shapes):
        check(row[0], expected, "transported field")
        check(row[1], expected, "tendency")
        operators._shared_profile(row[2], "spacing", nz)
        operators._cuda_float32(row[3], "map factor")
        if tuple(row[3].shape) != expected[2:]:
            raise ValueError("family map factor needs its target staggering")
    for value, expected, name in ((ru, shapes[1], "ru"), (rv, shapes[2], "rv"), (rw, shapes[3], "rw")):
        check(value, expected, name)
    for name, value in (("fnm", fnm), ("fnp", fnp)):
        operators._shared_profile(value, name, nz)
    if any(row[2].data.ptr != rows[0][2].data.ptr for row in rows[1:3]):
        raise ValueError("family scalar/u/v spacing must be one verified shared allocation")
    if rows[3][3].data.ptr != rows[0][3].data.ptr:
        raise ValueError("family scalar/w map factors must be one verified shared allocation")
    inputs = tuple((f"field {at}", row[0]) for at, row in enumerate(rows))
    inputs += (("ru", ru), ("rv", rv), ("rw", rw), ("fnm", fnm), ("fnp", fnp))
    inputs += tuple((f"spacing/map {at}/{part}", row[part]) for at, row in enumerate(rows) for part in (2, 3))
    for at, row in enumerate(rows):
        _separate(row[1], inputs + tuple((f"tendency {other}", value[1])
                                        for other, value in enumerate(rows) if other != at))
    if (open_x and nx < operators.FIFTH_ORDER_STENCIL_AXIS
            or open_y and ny < operators.FIFTH_ORDER_STENCIL_AXIS):
        raise ValueError("open family advection needs axis >= 7 so boundary degrade bands do not overlap")
    _scalar_index_limit(max(array.size // members for array in arrays))
    dx, dy = operators._finite(dx, "dx"), operators._finite(dy, "dy")
    if dx <= 0 or dy <= 0:
        raise ValueError("family advection spacing must be positive")
    if vorder_scalar not in (3, 5) or vorder_momentum not in (3, 5):
        raise ValueError("family advection vorder must be 3 or 5 (the WRF vert_order "
                         "ladders the kernel carries)")
    row_orders = {"": vorder_scalar, "x": vorder_momentum, "y": vorder_momentum,
                  "z": vorder_scalar}
    row_horders = {"": horder_scalar, "x": horder_momentum, "y": horder_momentum,
                   "z": horder_scalar}
    family_defines = operators.advection_order_defines(horder_scalar)
    if members > 1 and (operators.advection_order_defines(horder_momentum) != family_defines
                        or theta_offset):
        raise BatchKernelUnsupported(
            "the family translation unit compiles one horizontal order tier for all four "
            f"rows; scalar order {horder_scalar}, momentum order {horder_momentum}"
            f"{' and the theta offset row' if theta_offset else ''} need per-row entries "
            "(family=False), or the family would advect a row at another order than a single run")
    if members == 1:
        originals = tuple(prepare_flux_div_trial(
            row[0], ru, rv, rw, row[1], row[2], fnm, fnp, row[3], dx=dx, dy=dy,
            stagger=row[4], open_x=open_x, open_y=open_y, has_msf=has_msf,
            spec=specified, layout=layout, zero_tendency=False,
            vorder=row_orders[row[4]], horder=row_horders[row[4]],
            theta_offset=theta_offset and row[4] == "") for row in rows)
        scalar_zero = None
        if zero_tendency:
            from woof.ensemble.batch_bookkeeping import prepare_bookkeeping
            scalar_zero = prepare_bookkeeping(tuple(
                (_outer_view(row[1], layout), _outer_view(row[1], layout)) for row in rows),
                members=1, zero=True)
        def scalar_family():
            if scalar_zero is not None:
                scalar_zero()
            for original in originals:
                original()
        return _trial(scalar_family, candidate="advection_family", members=1,
                      layout=layout, n1_original_entries=True, family_kernel_launches_per_stage=0,
                      delegated_original_entries_per_stage=4,
                      delegated_zero_kernel_calls_per_stage=int(bool(zero_tendency)),
                      zero_tendency=bool(zero_tendency))
    from woof.ensemble.batch_kernel import _runtime_audit_options
    from woof.core.kernel_shape import shape_defines, specialize_source
    scalar_source = specialize_source("advection", _order_source(family_defines),
        shape_defines("advection", "flux_div_scalar", (nz, ny, nx)))
    source, spec = advection_family_source(
        scalar_source, members, layout=layout, zero_tendency=zero_tendency,
        audit_options=_runtime_audit_options(operators.FLUX_DIV_SCALAR_SPEC))
    values = arrays + (rows[0][2], rows[3][2], fnm, fnp,
                       rows[0][3], rows[1][3], rows[2][3])
    args = values + (operators._kernel_scalar(1.0 / dx, "inverse dx"),
                     operators._kernel_scalar(1.0 / dy, "inverse dy"),
                     np.int32(nz), np.int32(ny), np.int32(nx), np.int32(open_x),
                     np.int32(open_y), np.int32(has_msf), np.int32(specified),
                     np.int32(vorder_scalar), np.int32(vorder_momentum))
    inner = layout == "innermost"
    multiplier = members if inner else 1
    scalar_blocks = (nx * multiplier + _TPB - 1) // _TPB
    u_blocks = ((nx + 1) * multiplier + _TPB - 1) // _TPB
    grid = (3 * scalar_blocks + u_blocks, ny + 1, nz + 1)
    strides = {pointer.name: (0 if pointer.role == "shared" else value.strides[0])
               for pointer, value in zip(spec.pointers, values)}
    launch = prepare_batch_source_launch(
        source, spec, 1 if inner else members, grid, (_TPB, 1, 1), args,
        pointer_strides=None if inner else strides)
    return _trial(launch, candidate="advection_family_dispatch", members=members, layout=layout,
                  n1_original_entries=False, family_kernel_launches_per_stage=1,
                  original_entry_bodies_per_stage=4, zero_tendency=bool(zero_tendency),
                  source_sha256=sha256(source.encode()).hexdigest(),
                  kernel_attributes=_resources(source, spec, 1 if inner else members, launch),
                  byte_limit="family dispatch removes launches; zero fusion retains its separate address-byte bound")


def layout_trial_workspace_inventory(cfg, members, selection, *, allocation_quantum=512):
    """Logical words and separately rounded backing charges for inner stages."""
    from woof.ensemble.batch_kernel import _members
    from woof.ensemble.batch_storage import _positive
    if selection not in LAYOUT_TRIALS:
        raise ValueError("layout_trial must be none, outermost or innermost")
    members = _members(members)
    quantum = _positive(allocation_quantum, "allocation_quantum")
    if selection != "innermost" or members == 1:
        return ()
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    shapes = {"scalar": (nz, ny, nx), "u": (nz, ny, nx + 1),
              "v": (nz, ny + 1, nx), "w": (nz + 1, ny, nx)}
    fields = (("flux_ru", "u"), ("flux_rv", "v"), ("flux_ww", "w"),
              ("field_theta", "scalar"), ("field_u", "u"), ("field_v", "v"), ("field_w", "w"),
              ("tendency_theta", "scalar"), ("tendency_u", "u"), ("tendency_v", "v"), ("tendency_w", "w"))
    rows = []
    for name, staggering in fields:
        shape = shapes[staggering] + (members,)
        payload = int(np.prod(shape)) * 4
        rows.append({"name": name, "shape": shape, "payload_bytes": payload,
                     "allocated_bytes": ((payload + quantum - 1) // quantum) * quantum})
    return tuple(rows)


def layout_trial_workspace_bytes(cfg, members, selection, *, allocation_quantum=512):
    """Additional admission charge, rounded separately for each allocation."""
    return sum(row["allocated_bytes"] for row in layout_trial_workspace_inventory(
        cfg, members, selection, allocation_quantum=allocation_quantum))


def layout_trial_workspace_payload_bytes(cfg, members, selection):
    """Additional logical FP32 payload, excluding allocator block padding."""
    return sum(row["payload_bytes"] for row in layout_trial_workspace_inventory(cfg, members, selection))


def _order_source(defines):
    """The advection unit at the scalar launcher's horizontal order tier."""
    from woof.core.kernels import module_source, module_source_int_defines
    return (module_source_int_defines("advection", defines) if defines
            else module_source("advection"))


def prepare_stage_trials(state, ru, rv, ww, rows, *, layout, family=False,
                         horders=None, theta_offset=False):
    """Bind a complete stage trial, with all inner transitions inside launches.

    Authoritative state stays member-outermost. Inner stage fluxes are packed
    once per RK stage, then shared by its four stencil launches. Every field
    pack and tendency/Omega unpack executes inside the returned closures.
    Additional workspace is reported separately from the existing inventory.
    """
    if layout not in _LAYOUTS:
        raise ValueError("stage trial layout must be outermost or innermost")
    cfg = state.cfg
    rows = tuple(rows)
    if len(rows) != 4 or tuple(row[-1] for row in rows) != ("", "x", "y", "z"):
        raise ValueError("stage trial needs the scalar/u/v/w advection family in original order")
    options = dict(dx=cfg.dx, dy=cfg.dy, has_msf=state.has_msf)
    from woof.core.advection import vertical_orders
    vorder_scalar, vorder_momentum = vertical_orders(cfg)
    if horders is None:
        from woof.ensemble.batch_operators import flux_row_horizontal_orders
        horders = flux_row_horizontal_orders(cfg)
    family_orders = dict(vorder_scalar=vorder_scalar, vorder_momentum=vorder_momentum,
                         horder_scalar=horders[""], horder_momentum=horders["x"],
                         theta_offset=theta_offset)
    row_orders = {"": vorder_scalar, "x": vorder_momentum, "y": vorder_momentum,
                  "z": vorder_scalar}
    def row_tier(stagger):
        return dict(vorder=row_orders[stagger], horder=horders[stagger],
                    theta_offset=theta_offset and stagger == "")
    requested_layout = layout
    layout = "outermost" if state.members == 1 else layout
    additional = layout_trial_workspace_bytes(
        cfg, state.members, layout, allocation_quantum=state.plan.allocation_quantum)
    payload = layout_trial_workspace_payload_bytes(cfg, state.members, layout)
    metadata = {"selection": requested_layout, "additional_workspace_bytes": additional,
                "additional_workspace_payload_bytes": payload,
                "allocation_quantum": state.plan.allocation_quantum,
                "fused_zero_fields": ("rth_t", "ru_t", "rv_t", "rw_t"),
                "advection_family_dispatch": bool(family),
                "state_layout": "outermost", "transitions_in_step": layout == "innermost"}
    if layout == "outermost":
        omega = prepare_omega_trial(ru, rv, ww, state.dnw, state.c1h,
                                    msft=state.msft, layout=layout, **options)
        if family:
            advection = (prepare_advection_family_trial(
                rows, ru, rv, ww, state.fnm, state.fnp, layout=layout,
                zero_tendency=True, **family_orders, **options),)
        else:
            advection = tuple(prepare_flux_div_trial(
                field, ru, rv, ww, tendency, spacing, state.fnm, state.fnp, msf,
                stagger=stagger, layout=layout, zero_tendency=True,
                **row_tier(stagger), **options)
                for field, tendency, spacing, msf, stagger in rows)
    else:
        import cupy as cp
        available, _ = cp.cuda.runtime.memGetInfo()
        reserve = state.plan.reserved_bytes
        if additional + reserve > available:
            raise MemoryError(
                f"inner stage buffers need {additional} bytes plus {reserve} reserve; "
                f"only {available} CUDA bytes are free, so allocation would exceed admission")
        def empty_inner(array):
            return cp.empty(tuple(array.shape[1:]) + (state.members,), dtype=cp.float32)
        pool = cp.get_default_memory_pool()
        before_buffers = pool.used_bytes()
        inner_ru, inner_rv, inner_ww = tuple(empty_inner(value) for value in (ru, rv, ww))
        inner_fields = tuple(empty_inner(row[0]) for row in rows)
        inner_tendencies = tuple(empty_inner(row[1]) for row in rows)
        buffers = (inner_ru, inner_rv, inner_ww) + inner_fields + inner_tendencies
        buffer_charge = pool.used_bytes() - before_buffers
        metadata["observed_buffer_pool_charge_bytes"] = buffer_charge
        if sum(value.nbytes for value in buffers) != payload:
            raise RuntimeError("inner stage allocation differs from its exact quoted word inventory")
        if buffer_charge > additional:
            raise RuntimeError("inner stage allocator charge exceeds its rounded admission quote")
        inner_omega = prepare_omega_trial(
            inner_ru, inner_rv, inner_ww, state.dnw, state.c1h, msft=state.msft,
            layout=layout, cache_shared=True, **options)
        def omega():
            cp.copyto(inner_ru, cp.moveaxis(ru, 0, -1))
            cp.copyto(inner_rv, cp.moveaxis(rv, 0, -1))
            inner_omega()
            unpack_member_innermost(inner_ww, out=ww)
        omega.metadata = inner_omega.metadata
        if family:
            inner_rows = tuple((packed, output, row[2], row[3], row[4])
                               for row, packed, output in zip(rows, inner_fields, inner_tendencies))
            inner_family = prepare_advection_family_trial(
                inner_rows, inner_ru, inner_rv, inner_ww, state.fnm, state.fnp,
                layout=layout, zero_tendency=True, **family_orders, **options)
            def transported_family():
                for row, packed in zip(rows, inner_fields):
                    cp.copyto(packed, cp.moveaxis(row[0], 0, -1))
                inner_family()
                for row, packed_output in zip(rows, inner_tendencies):
                    unpack_member_innermost(packed_output, out=row[1])
            transported_family.metadata = inner_family.metadata
            advection = (transported_family,)
        else:
            advection = []
            for row, field_inner, tendency_inner in zip(rows, inner_fields, inner_tendencies):
                field, tendency, spacing, msf, stagger = row
                inner_launch = prepare_flux_div_trial(
                    field_inner, inner_ru, inner_rv, inner_ww, tendency_inner,
                    spacing, state.fnm, state.fnp, msf, stagger=stagger,
                    layout=layout, zero_tendency=True,
                    **row_tier(stagger), **options)
                def transported(operation=inner_launch, source=field, packed=field_inner,
                                packed_output=tendency_inner, destination=tendency):
                    cp.copyto(packed, cp.moveaxis(source, 0, -1))
                    operation()
                    unpack_member_innermost(packed_output, out=destination)
                transported.metadata = inner_launch.metadata
                advection.append(transported)
            advection = tuple(advection)
        metadata.update(conversion_copy_calls_per_stage=11,
                        conversion_bytes_per_stage=4 * state.members * (
                            (ru.size + rv.size + ww.size) // state.members
                            + sum((row[0].size + row[1].size) // state.members for row in rows)) * 2)
    metadata["omega"] = dict(omega.metadata)
    metadata["advection"] = tuple(dict(launch.metadata) for launch in advection)
    return omega, advection, metadata
