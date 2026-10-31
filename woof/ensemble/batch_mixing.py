"""Prepared dry metric km_opt=4 and sixth-order forward tendencies.

Existing CUDA bodies, scalar conversion trees and CuPy ufuncs are retained.
Each arithmetic entry executes once over all independent member slabs. This
binds held tendencies; it does not integrate a forecast or attach physics.
"""
from __future__ import annotations

import re
from functools import lru_cache
from hashlib import sha256

import numpy as np

from woof.core.device_inventory import state_array_shapes
from woof.core.device_cache import cuda_cache
from woof.ensemble.batch_kernel import (
    KernelSpec, PointerSpec, _entry_parts, _runtime_audit_options,
    _argument_owners, _finish_prepared_launch, batch_grid, pack_pointer_strides,
    prepare_batch_kernel_launch, validate_pointer_arguments,
)
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported, _exact_key
from woof.ensemble.batch_storage import BatchArraySpec

_TPB = 128
_F = np.float32
_COMMON_FIELDS = (
    ("u", "u0"), ("v", "v0"), ("w", "w0"), ("php", "php0"),
    ("phb", "phb"), ("alt", "alt"), ("qv", "alt"),
    ("msft", "msft"), ("msfu", "msfu"), ("msfv", "msfv"),
    ("fnm", "fnm"), ("fnp", "fnp"), ("dn", "dn"), ("dnw", "dnw"),
)
_GRID_SIGNATURE = (
    "const real* u, const real* v, const real* w, const real* php, "
    "const real* phb, const real* alt, const real* qv, const real* msft, "
    "const real* msfu, const real* msfv, const real* fnm, const real* fnp, "
    "const real* dn, const real* dnw, real rdx, real rdy, real dx, real dy, "
    "real cf1, real cf2, real cf3, int moist"
)
_GRID_MACRO_ENTRIES = frozenset({
    "wrf_smag_hd_u", "wrf_smag_hd_v", "wrf_smag_vd_u", "wrf_smag_vd_v", "wrf_smag_vd_w",
    "wrf_smag_surface_u", "wrf_smag_surface_v", "wrf_smag_surface_scalars", "wrf_smag_hd_w",
    "wrf_smag_hd_w_cached", "wrf_smag_w_primitives", "wrf_smag_hd_w_stress", "wrf_smag_w_stress",
    "wrf_smag_flux_s", "wrf_smag_hd_s", "wrf_calc_n2", "wrf_smag3d_km", "wrf_smag_vd_s",
    "wrf_smag_surface_u_cd0", "wrf_smag_surface_v_cd0", "wrf_smag_surface_heat_const",
    "wrf_tke_km", "wrf_tke_rhs",
})
_SMAG_PTX_ENTRIES = _GRID_MACRO_ENTRIES | {"wrf_smag_deform", "wrf_smag2d_km", "wrf_smag_km_bc"}
_ROWS = (("u0", "smag_ru", "diff6_x", "x", "c1h", "c2h"),
         ("v0", "smag_rv", "diff6_y", "y", "c1h", "c2h"),
         ("w0", "smag_rw", "diff6_z", "z", "c1f", "c2f"),
         ("thp0", "smag_rth", "diff6_m", "", "c1h", "c2h"))


@lru_cache(maxsize=None)
def _scalar_smag_ptx(audit_options):
    """Expose the original frontend's PTX with its original arithmetic flags.

    The real architecture is retained even when requesting the intermediate
    PTX. Source and option hashes distinguish the NVRTC program cache key.
    This is used by the compiled-arithmetic identity proof before batching.
    """
    from cupy.cuda import compiler
    from woof.core.kernels import module_options, module_source
    from woof import wrf_exact
    source = module_source("smag2d")
    options = module_options("smag2d")
    if wrf_exact.ENABLED:
        options = wrf_exact.effective_options(options)
    options += (audit_options[-1], "--device-as-default-execution-space")
    if getattr(compiler, "_use_pch", False):
        options += ("--pch",)
    key = sha256((source + repr(options)).encode()).hexdigest()
    program = compiler._NVRTCProgram(source, name=key + ".cu", method="ptx")
    ptx, _ = program.compile(options)
    if isinstance(ptx, bytes):
        ptx = ptx.decode("utf-8")
    build = re.search(r"\bV(\d+\.\d+\.\d+)\b", ptx)
    return ptx, {"source_sha256": sha256(source.encode()).hexdigest(),
                 "scalar_ptx_sha256": sha256(ptx.encode()).hexdigest(),
                 "nvrtc_version": compiler._get_nvrtc_version(),
                 "nvrtc_build": build[1] if build else "not exposed in PTX header",
                 "scalar_nvrtc_options": options}


def _ptx_to_cubin(ptx, architecture):
    """Assemble PTX with the installed NVRTC version's CPU linker.

    A new driver is not needed to parse the frontend's PTX ISA. No arithmetic
    option is added: the original PTX carries its rounding and contractions.
    nvJitLink's default split compilation uses one CPU thread.
    """
    import ctypes as c
    import ctypes.util
    import importlib.metadata
    from cupy.cuda import compiler
    version = tuple(compiler._get_nvrtc_version())
    distributions = []
    for package in ("nvidia-nvjitlink", "nvidia-nvjitlink-cu12"):
        try:
            distributions.append(importlib.metadata.distribution(package))
        except importlib.metadata.PackageNotFoundError:
            pass
    paths = [(distribution.locate_file(path), distribution.version)
             for distribution in distributions for path in distribution.files or ()
             if re.search(r"(?:libnvJitLink\.so(?:\.\d+)?|nvJitLink[^/]*\.dll)$", str(path))]
    system_library = ctypes.util.find_library("nvJitLink")
    candidates = paths + ([(system_library, "system library")] if system_library else [])
    if not candidates:
        raise BatchStateUnsupported("metric PTX batching needs the installed NVRTC version's nvJitLink to retain the scalar floating graph")

    lib = None
    package_version = None
    for library, candidate_version in candidates:
        try:
            candidate = c.CDLL(str(library))
            get_version = candidate.nvJitLinkVersion
            get_version.argtypes = [c.POINTER(c.c_uint), c.POINTER(c.c_uint)]
            get_version.restype = c.c_int
            major, minor = c.c_uint(), c.c_uint()
            if not get_version(c.byref(major), c.byref(minor)) and (major.value, minor.value) == version:
                lib, package_version = candidate, candidate_version
                break
        except (OSError, AttributeError):
            continue
    if lib is None:
        raise BatchStateUnsupported("no installed nvJitLink matches NVRTC; mixed compiler versions can change scalar rounding")

    def api(name, types):
        function = getattr(lib, name)
        function.argtypes, function.restype = types, c.c_int
        return function

    create = api("nvJitLinkCreate", [c.POINTER(c.c_void_p), c.c_uint, c.POINTER(c.c_char_p)])
    destroy = api("nvJitLinkDestroy", [c.POINTER(c.c_void_p)])
    add = api("nvJitLinkAddData", [c.c_void_p, c.c_int, c.c_void_p, c.c_size_t, c.c_char_p])
    complete = api("nvJitLinkComplete", [c.c_void_p])
    get_size = api("nvJitLinkGetLinkedCubinSize", [c.c_void_p, c.POINTER(c.c_size_t)])
    get_data = api("nvJitLinkGetLinkedCubin", [c.c_void_p, c.c_void_p])
    error_size = api("nvJitLinkGetErrorLogSize", [c.c_void_p, c.POINTER(c.c_size_t)])
    error_data = api("nvJitLinkGetErrorLog", [c.c_void_p, c.c_void_p])
    handle = c.c_void_p()
    option = architecture.replace("-arch=compute_", "-arch=sm_").encode()
    options = (c.c_char_p * 1)(option)
    if create(c.byref(handle), 1, options):
        raise RuntimeError("nvJitLink could not create the original-architecture compiler")

    def check(code):
        if code:
            length = c.c_size_t()
            error_size(handle, c.byref(length))
            error = c.create_string_buffer(max(1, length.value))
            error_data(handle, error)
            raise RuntimeError(f"nvJitLink code {code}: {error.value.decode('utf-8')}")

    try:
        data = c.create_string_buffer(ptx.encode())
        check(add(handle, 2, data, len(data), b"scalar-arithmetic.ptx"))
        check(complete(handle))
        length = c.c_size_t()
        check(get_size(handle, c.byref(length)))
        linked = c.create_string_buffer(length.value)
        check(get_data(handle, linked))
        return linked.raw, {"nvjitlink_version": version, "nvjitlink_options": (option.decode(),),
                            "nvjitlink_package_version": package_version,
                            "cubin_sha256": sha256(linked.raw).hexdigest()}
    finally:
        destroy(c.byref(handle))


@cuda_cache(maxsize=None)
def _original_smag_ptx_module(audit_options):
    """Load scalar PTX through its matching linker for the raw identity gate."""
    from cupy.cuda import function
    ptx, receipt = _scalar_smag_ptx(audit_options)
    cubin, link_receipt = _ptx_to_cubin(ptx, audit_options[-1])
    module = function.Module()
    module.load(cubin)
    return module, {**receipt, **link_receipt}


def _original_smag_ptx_kernel(entry, audit_options):
    module, receipt = _original_smag_ptx_module(audit_options)
    return module.get_function(entry), receipt


def _member_smag_ptx(ptx, spec, members, parameter_names):
    """Bind member coordinates and addresses after scalar FP contraction.

    Source-level pointer rebinding changes NVRTC's choice of which product a
    sum fuses, including stress interpolation and pair differences. This
    adapter retains every original floating instruction and branch in PTX.
    Its added instructions are integer coordinate and byte-address arithmetic.
    Only this audited Smagorinsky module is admitted at this boundary.
    """
    from woof.ensemble.batch_kernel import _close, _masked
    if spec.module != "smag2d" or spec.entry not in _SMAG_PTX_ENTRIES:
        raise BatchStateUnsupported("compiled metric mixing rebinding needs an audited installed module entry")
    if members < 2 or isinstance(members, bool):
        raise ValueError("compiled metric member rebinding requires at least two members")
    masked = _masked(ptx)
    declarations = list(re.finditer(r"\.visible\s+\.entry\s+" + re.escape(spec.entry) + r"\s*\(", masked))
    if len(declarations) != 1:
        raise BatchStateUnsupported("metric PTX entry is missing or duplicated")
    start = declarations[0].start()
    open_param = declarations[0].end() - 1
    close_param = _close(masked, open_param, "(", ")")
    open_body = masked.index("{", close_param)
    close_body = _close(masked, open_body, "{", "}")
    signature = ptx[open_param + 1:close_param]
    parameters = [part.strip() for part in signature.split(",")]
    if len(parameters) != len(parameter_names):
        raise BatchStateUnsupported("scalar PTX parameter count differs from the audited CUDA ABI")
    ptx_names = []
    pointer_names = {pointer.name for pointer in spec.pointers}
    for name, parameter in zip(parameter_names, parameters):
        # NVRTC annotates pointer parameters with ".ptr .align N" for some
        # targets (sm_120) and not others (sm_89); the audited ABI is the
        # type class: a pointer is a .u64 parameter, a scalar is .f32 or .u32,
        # and each must agree with the audited pointer roster by name.
        match = re.fullmatch(r"\.param\s+(\.u64(?:\s+\.ptr\s+\.align\s+\d+)?|\.(?:f32|u32))\s+([A-Za-z_]\w*)", parameter)
        if match is None or (name in pointer_names) != match[1].startswith(".u64"):
            raise BatchStateUnsupported("scalar PTX parameter types differ from the audited float32/pointer ABI")
        ptx_names.append(match[2])
    body = ptx[open_body + 1:close_body]
    if re.search(r"\bcall\b|%ensemble_", _masked(body)):
        raise BatchStateUnsupported("metric PTX gained a device call or reserved rebinding register; its dataflow needs an audit")
    if re.search(r"%nctaid\.[yz]", body):
        raise BatchStateUnsupported("metric PTX gained a non-x grid-dimension dependency; its virtual grid needs an audit")
    body = body.replace("%ctaid.x", "%ensemble_block_x").replace("%nctaid.x", "%ensemble_grid_x")
    descriptor = spec.entry + "_member_strides"
    for index, pointer in enumerate(spec.pointers):
        if pointer.role == "shared":
            continue
        parameter = ptx_names[parameter_names.index(pointer.name)]
        pattern = re.compile(r"(?m)^(\s*ld\.param\.(?:b64|u64)\s+(%rd\d+),\s*\[" + re.escape(parameter) + r"\];)\s*$")
        reads = list(pattern.finditer(body))
        if len(reads) != len(re.findall(r"\b" + re.escape(parameter) + r"\b", body)):
            raise BatchStateUnsupported("metric PTX pointer parameter gained an unaudited load or direct address use")

        def offset(match):
            register = match[2]
            return (match[1] + "\n"
                    f"\tld.param.u64 %ensemble_stride, [{descriptor}+{index * 8}];\n"
                    "\tmul.lo.u64 %ensemble_offset, %ensemble_member64, %ensemble_stride;\n"
                    f"\tadd.u64 {register}, {register}, %ensemble_offset;")
        body = pattern.sub(offset, body)
    declarations = re.match(r"(?:\s|//[^\n]*\n|\.reg[^;]*;)*", body)
    if declarations is None:
        raise BatchStateUnsupported("metric PTX register declaration layout changed")
    registers = ("\n\t.reg .b32 %ensemble_physical_x, %ensemble_physical_grid_x, %ensemble_grid_x, %ensemble_member, %ensemble_block_x;\n"
                 "\t.reg .b64 %ensemble_member64, %ensemble_stride, %ensemble_offset;\n")
    setup = ("\tmov.u32 %ensemble_physical_x, %ctaid.x;\n"
             "\tmov.u32 %ensemble_physical_grid_x, %nctaid.x;\n"
             f"\tdiv.u32 %ensemble_grid_x, %ensemble_physical_grid_x, {members};\n"
             "\tdiv.u32 %ensemble_member, %ensemble_physical_x, %ensemble_grid_x;\n"
             "\trem.u32 %ensemble_block_x, %ensemble_physical_x, %ensemble_grid_x;\n"
             "\tcvt.u64.u32 %ensemble_member64, %ensemble_member;\n")
    body = registers + body[:declarations.end()] + setup + body[declarations.end():]
    before = [line.strip() for line in ptx[open_body + 1:close_body].splitlines() if re.search(r"\.(?:f32|f64)\b", line)]
    after = [line.strip() for line in body.splitlines() if re.search(r"\.(?:f32|f64)\b", line)]
    if before != after:
        raise AssertionError("member PTX adapter changed the scalar floating instruction sequence")
    # Keep the original header and globals. Every helper is inlined, as audited
    # above; unrelated global entries need not be reassembled for this handle.
    first_entry = re.search(r"\.visible\s+\.entry\b", masked).start()
    signature += f",\n\t.param .align 8 .b8 {descriptor}[{len(spec.pointers) * 8}]\n"
    result = ptx[:first_entry] + ptx[start:open_param + 1] + signature + ")\n{" + body + "}\n"
    return result, sha256("\n".join(before).encode()).hexdigest()


@cuda_cache(maxsize=None)
def _member_smag_kernel(spec, members, audit_options):
    from cupy.cuda import function
    from types import SimpleNamespace
    from woof.certify.kernel_manifest import record_module
    ptx, receipt = _scalar_smag_ptx(audit_options)
    names = _entry_parts(normalized_smag_source(), spec, audit_options)[-2]
    adapted, float_hash = _member_smag_ptx(ptx, spec, members, names)
    cubin, link_receipt = _ptx_to_cubin(adapted, audit_options[-1])
    module = function.Module()
    module.load(cubin)
    key = f"woof.ensemble.batch_mixing:{spec.entry}[members={members}]"
    record_module(key, source=adapted, options=link_receipt["nvjitlink_options"],
                  module=SimpleNamespace(cubin=cubin))
    receipt = {**receipt, **link_receipt, "source_kind": "scalar-ptx-member-addresses",
               "compiled_ptx_sha256": sha256(adapted.encode()).hexdigest(),
               "floating_instructions_sha256": float_hash}
    return module.get_function(spec.entry), receipt, {"scalar.ptx": ptx,
                                                     "member.ptx": adapted,
                                                     "member.cubin": cubin}


def _prepare_member_smag_launch(state, spec, grid, args, strides):
    from woof.ensemble.batch_kernel import _current_device
    audit_options = _runtime_audit_options(spec)
    names = _entry_parts(normalized_smag_source(), spec, audit_options)[-2]
    device = _current_device()
    owners, arrays = _argument_owners(args, device)
    validate_pointer_arguments(spec, state.members, args, strides, names)
    descriptor = pack_pointer_strides(spec, strides)
    kernel, receipt, artifacts = _member_smag_kernel(spec, state.members, audit_options)
    launch = _finish_prepared_launch(spec, state.members, device, batch_grid(grid, state.members),
                                    (_TPB, 1, 1), tuple(args) + (descriptor,), kernel,
                                    owners, arrays, audit_options, 0, None, receipt)
    launch.compiler_artifacts = artifacts
    return launch


def _require_wrf461_diff6(cfg):
    """Decline the fork's sixth-order filter form on the batched graph.

    Its launches bind WRF v4.6.1's single factor, dt/3 scalar step and
    three-point edge mask (``_diff6_launch``); a member configured with
    ``diff_6th_form = "noaa_wrf39"`` would run a filter nine times
    stronger on its moisture than it asked for.  The ordinary door runs
    that form (woof.core.dycore.prepare_fixed_tendencies).
    """
    if cfg.diff_6th_opt > 0 and getattr(cfg, "diff_6th_form", "wrf_461") != "wrf_461":
        raise BatchStateUnsupported(
            "the batched mixing graph binds the WRF v4.6.1 sixth-order filter; "
            f"diff_6th_form = {cfg.diff_6th_form!r} runs on the ordinary door")


def workspace_specs(cfg, *, has_msf=True):
    """Declare every additional backing before prepared host input admission.

    ``has_msf`` must match prepared scalar metadata. The reciprocal is needed
    only by the original big-step v map branch. Existing scratch shapes come
    from scratch_slot_registry through BatchedDomainState, without duplication.
    """
    from woof.core.dycore import BIGSTEP_ENABLED
    if cfg.km_opt not in (0, 1, 4) or (cfg.km_opt == 4 and cfg.diff_opt != 2):
        raise BatchStateUnsupported("dry mixing binds km_opt=4/diff_opt=2 or diff6-only; km_opt=2/3 need separate closure bindings")
    _require_wrf461_diff6(cfg)
    if cfg.km_opt != 4 and cfg.diff_6th_opt <= 0:
        return ()
    shapes = state_array_shapes(cfg)
    result = [BatchArraySpec("mixing_mut", shapes["mup"], "member")]
    if has_msf and BIGSTEP_ENABLED:
        result.append(BatchArraySpec("mixing_inv_msfv", shapes["msfv"], "member"))
    return tuple(result)


def required_scratch_slots(cfg):
    """Request existing registry slots by name, never restate their shapes."""
    from woof.core.preflight import scratch_slot_registry
    workspace_specs(cfg, has_msf=False)
    slots = set()
    if cfg.km_opt == 4 or cfg.diff_6th_opt > 0:
        slots.update(row[1] for row in _ROWS)
        slots.update(("diff6_x", "diff6_y"))
    if cfg.km_opt == 4:
        slots.update(("smag_km", "smag_kh"))
        # The original non-sm_120 route borrows step-dead acoustic slots
        # for its cached W geometry. These remain the registry's shapes.
        slots.update(("acoustic_a", "acoustic_c2a"))
    if cfg.diff_6th_opt > 0:
        slots.update(("diff6_z", "diff6_m"))
    registry = scratch_slot_registry(cfg)
    if slots - registry.keys():
        raise BatchStateUnsupported("mixing scratch registry changed; audit its named workspace inventory")
    return {name: np.dtype("float32") for name in sorted(slots)}


def normalized_smag_source():
    """Expand one checked argument macro only, leaving all CUDA bodies intact.

    The original loader keeps its source at N=1. The generic batch adapter
    requires explicit pointer parameters, so this seam exposes the installed
    object macro's existing tokens in global signatures before N>1 binding.
    """
    from woof.core.kernels import module_source
    source = module_source("smag2d")
    matches = list(re.finditer(r"(?m)^#define WRF_SMAG_GRID_ARGS\s+((?:[^\n]*\\\n)*[^\n]*)", source))
    if len(matches) != 1:
        raise BatchStateUnsupported("WRF_SMAG_GRID_ARGS must have one audited object macro definition")
    expansion = re.sub(r"\\\n", " ", matches[0][1])
    if " ".join(expansion.split()) != _GRID_SIGNATURE:
        raise BatchStateUnsupported("WRF_SMAG_GRID_ARGS pointer/scalar contract changed; rebinding requires an explicit ABI audit")
    pattern = re.compile(r"(\bvoid\s+(\w+)\s*\(\s*)WRF_SMAG_GRID_ARGS\b")
    names = tuple(match[2] for match in pattern.finditer(source))
    if len(names) != len(set(names)) or set(names) != _GRID_MACRO_ENTRIES:
        raise BatchStateUnsupported("metric Smagorinsky signature inventory changed; audit every macro entry before batching")
    result, count = pattern.subn(lambda match: match[1] + expansion, source)
    if count != len(_GRID_MACRO_ENTRIES):
        raise AssertionError("checked metric signature expansion omitted an entry")
    return result


def _array(state, name, *, output=False):
    spec = state.storage.specs.get(name)
    if spec is None:
        raise BatchStateUnsupported(f"dry mixing requires planned allocation {name!r}; an implicit temporary would bypass admission")
    value = state.storage.arrays[name]
    if (np.dtype(spec.dtype) != np.dtype("float32") or value.dtype != np.dtype("float32")
            or tuple(value.shape) != spec.allocation_shape(state.members)
            or not value.flags.c_contiguous or not hasattr(value, "__cuda_array_interface__")):
        raise ValueError(f"mixing allocation {name!r} differs from its admitted float32 slab")
    if output and spec.ownership != "member":
        raise BatchStateUnsupported(f"mixing output {name!r} must be independently owned by each member")
    return value


def _prefix(state, slot):
    """A mass-sized prefix of EACH member's original padded scratch slab."""
    name = "scratch:" + slot
    value = _array(state, name, output=True)
    shape = state.storage.specs["p"].shape
    count = int(np.prod(shape))
    result = value.reshape(state.members, -1)[:, :count].reshape((state.members,) + shape)
    if (result.dtype != value.dtype or result.data.ptr != value.data.ptr
            or result.data.mem is not value.data.mem):
        raise BatchStateUnsupported("mixing tensor prefix did not retain its admitted per-member backing")
    # A singleton leading axis has no adjacent member to address. CuPy may
    # replace that unused stride with the smaller logical-prefix size during
    # reshape. Its scalar view is valid only if the pointer, owner, inner
    # contiguity and complete logical span still prove the original backing.
    expected = result.dtype.itemsize
    for extent, stride in zip(reversed(shape), reversed(result.strides[1:])):
        if stride != expected:
            raise BatchStateUnsupported("mixing tensor prefix lost contiguous scalar inner axes")
        expected *= extent
    if state.members > 1 and result.strides[0] != value.strides[0]:
        raise BatchStateUnsupported("mixing tensor prefix changed the admitted member byte stride")
    memory = result.data.mem
    span = (state.members - 1) * result.strides[0] + count * result.dtype.itemsize
    if (result.data.ptr < memory.ptr or result.data.ptr + span > memory.ptr + memory.size):
        raise BatchStateUnsupported("mixing tensor prefix exceeds its admitted allocation bounds")
    return result


def _raw(state, module, entry, fields, args, grid):
    """Prepare original handles at N=1 and checked member-local raw handles."""
    spec = KernelSpec(module, entry, tuple(
        PointerSpec(parameter, state.storage.specs[name].ownership)
        for parameter, name in fields))
    strides = {parameter: state.storage.pointer_stride_bytes(name)
               for parameter, name in fields}
    if state.members == 1:
        # Preserve scalar source, options, dimensions and argument ordering.
        scalar_args = tuple(value[0] if hasattr(value, "__cuda_array_interface__")
                            and state.storage.specs[dict(fields)[parameter]].ownership == "member"
                            else value
                            for parameter, value in _named_arguments(module, entry, spec, args))
        launch = prepare_batch_kernel_launch(spec, 1, grid, (_TPB, 1, 1), scalar_args)
    elif module != "smag2d":
        launch = prepare_batch_kernel_launch(spec, state.members, grid, (_TPB, 1, 1),
                                             args, pointer_strides=strides)
    else:
        launch = _prepare_member_smag_launch(state, spec, grid, args, strides)
    launch.numerical_entries = (entry,)
    return launch


def _named_arguments(module, entry, spec, args):
    from woof.core.kernels import module_source
    source = normalized_smag_source() if module == "smag2d" else module_source(module)
    names = _entry_parts(source, spec, _runtime_audit_options(spec))[-2]
    if len(names) != len(args):
        raise ValueError(f"mixing argument count differs from {entry}'s installed signature")
    return tuple(zip(names, args))


def _grid(shape):
    nlev, nys, nxs = shape
    return ((nxs + _TPB - 1) // _TPB, nys, nlev)


def _context(state):
    from woof.core.dycore import BIGSTEP_ENABLED
    if not isinstance(state, BatchedDomainState):
        raise TypeError("prepared dry mixing needs an admitted BatchedDomainState")
    cfg = state.cfg
    if cfg.km_opt == 4 and (cfg.moist or state.physics is not None) and cfg.bl_pbl_physics == 0:
        raise BatchStateUnsupported("moist/physics surface mixing without a PBL needs admitted surface flux inputs; dry placeholders would omit those fluxes")
    if cfg.km_opt not in (0, 1, 4) or (cfg.km_opt == 4 and cfg.diff_opt != 2):
        raise BatchStateUnsupported("dry mixing supports km_opt=4/diff_opt=2 or diff6-only; km_opt=2/3 and diff_opt=1 are not bound")
    if cfg.khdif > 0 or cfg.kvdif > 0:
        raise BatchStateUnsupported("constant second-order diffusion requires its own member binding")
    _require_wrf461_diff6(cfg)
    if cfg.isfflx not in (0, 1, 2):
        raise ValueError("dry metric surface flux requires isfflx=0, 1 or 2")
    return cfg, BIGSTEP_ENABLED


def _common_fields(state):
    return tuple((parameter, "qv0" if parameter == "qv" and state.cfg.moist else name)
                 for parameter, name in _COMMON_FIELDS)


def _common(state):
    cfg = state.cfg
    values = [_array(state, name) for _, name in _common_fields(state)]
    values += [_F(1.0 / cfg.dx), _F(1.0 / cfg.dy), _F(cfg.dx), _F(cfg.dy),
               _F(state.cf1), _F(state.cf2), _F(state.cf3), np.int32(cfg.moist)]
    from woof.core.dycore import _boundary_x, _boundary_y
    dims = tuple(np.int32(value) for value in (
        cfg.nz, cfg.ny, cfg.nx, len(state.storage.specs["phb"].shape) == 3,
        _boundary_x(cfg), _boundary_y(cfg)))
    return tuple(values), dims


def _zero_strips(value, cfg, width):
    from woof.core.dycore import _zero_open_strips
    # This established slice helper accepts arbitrary leading axes. Every
    # assignment spans all members, with exactly the scalar strip bounds.
    _zero_open_strips(value, cfg, width)


def metric_w_entry_family(*, exact, compute_capability):
    """Select the compiled WRF w route independently of architecture.

    ``compute_capability`` no longer selects a route (xnode-identity,
    2026-10-04): a per-architecture choice broke cross-card byte identity.
    """
    if exact:
        return ("wrf_smag_hd_w",)
    return ("wrf_smag_w_stress", "wrf_smag_hd_w_stress")


def _smag_w_launches(state, common, dims, km, tend, fx, fy):
    from woof.core.dycore import WRF_EXACT, DIFFUSION_ENABLED
    cfg = state.cfg
    entries = metric_w_entry_family(exact=WRF_EXACT or DIFFUSION_ENABLED,
        compute_capability=tend.device.compute_capability)
    fields = _common_fields(state)
    grid = _grid(state.storage.specs["w0"].shape)
    if entries == ("wrf_smag_hd_w",):
        return (_raw(state, "smag2d", entries[0], fields +
            (("km", "scratch:smag_km"), ("tend", "scratch:smag_rw")),
            common + (km, tend) + dims, grid),)
    if entries[0] == "wrf_smag_w_stress":
        stress_grid = ((cfg.nx + 1 + _TPB - 1) // _TPB, cfg.ny + 1, cfg.nz)
        flux_fields = (("tx", "scratch:diff6_x"), ("ty", "scratch:diff6_y"))
        return (
            _raw(state, "smag2d", entries[0], fields + (("km", "scratch:smag_km"),) + flux_fields,
                 common + (km, fx, fy) + dims, stress_grid),
            _raw(state, "smag2d", entries[1], fields + flux_fields + (("tend", "scratch:smag_rw"),),
                 common + (fx, fy, tend) + dims, grid),
        )
    what = _array(state, "scratch:acoustic_a", output=True)
    rdz = _array(state, "scratch:acoustic_c2a", output=True)
    cache_fields = (("cached_what", "scratch:acoustic_a"), ("cached_rdz", "scratch:acoustic_c2a"),
                    ("cached_zx", "scratch:diff6_x"), ("cached_zy", "scratch:diff6_y"))
    cache = (what, rdz, fx, fy)
    primitive_grid = ((cfg.nx + 1 + _TPB - 1) // _TPB, cfg.ny + 1, cfg.nz + 1)
    return (
        _raw(state, "smag2d", entries[0], fields + cache_fields, common + cache + dims, primitive_grid),
        _raw(state, "smag2d", entries[1], fields + (("km", "scratch:smag_km"),) + cache_fields +
             (("tend", "scratch:smag_rw"),), common + (km,) + cache + (tend,) + dims, grid),
    )


def _smag_launches(state):
    """Bind the original metric closure, including dry surface-flux branches."""
    import cupy as cp
    from woof.core.dycore import _PRANDTL, _boundary_x, _boundary_y
    cfg = state.cfg
    common, dims = _common(state)
    mass_shape = state.storage.specs["p"].shape
    km, kh = (_array(state, "scratch:" + slot, output=True) for slot in ("smag_km", "smag_kh"))
    d11, d22, d12 = (_prefix(state, slot) for slot in ("smag_rw", "smag_ru", "smag_rv"))
    tensor_fields = (("d11", "scratch:smag_rw"), ("d22", "scratch:smag_ru"),
                     ("d12", "scratch:smag_rv"))
    launches = [_raw(state, "smag2d", "wrf_smag_deform", _common_fields(state) + tensor_fields,
                     common + (d11, d22, d12) + dims, _grid(mass_shape)),
                _raw(state, "smag2d", "wrf_smag2d_km", _common_fields(state) +
                     (("d11a", "scratch:smag_rw"), ("d22a", "scratch:smag_ru"),
                      ("d12a", "scratch:smag_rv"), ("xkmh", "scratch:smag_km"),
                      ("xkhh", "scratch:smag_kh")),
                     common + (_F(cfg.c_s), _F(_PRANDTL), d11, d22, d12, km, kh) + dims,
                     _grid(mass_shape))]
    if _boundary_x(cfg) or _boundary_y(cfg):
        launches.append(_raw(state, "smag2d", "wrf_smag_km_bc",
                             (("xkmh", "scratch:smag_km"), ("xkhh", "scratch:smag_kh")),
                             (km, kh) + dims[:3] + dims[4:], _grid(mass_shape)))
    first = tuple(launches)
    stresses = []
    for axis, slot, tensors in (
            ("u", "diff6_x", (("d11", "scratch:smag_rw", d11), ("d12", "scratch:smag_rv", d12))),
            ("v", "diff6_y", (("d22", "scratch:smag_ru", d22), ("d12", "scratch:smag_rv", d12)))):
        temporary = _array(state, "scratch:" + slot, output=True)
        fields = _common_fields(state) + (("km", "scratch:smag_km"),) + tuple(
            (parameter, name) for parameter, name, _ in tensors) + (("tend", "scratch:" + slot),)
        launch = _raw(state, "smag2d", "wrf_smag_hd_" + axis, fields,
                      common + (km,) + tuple(value for _, _, value in tensors) + (temporary,) + dims,
                      _grid(state.storage.specs["scratch:" + slot].shape))
        stresses.append((temporary, launch))
        launches.append(launch)
    rw = _array(state, "scratch:smag_rw", output=True)
    rth = _array(state, "scratch:smag_rth", output=True)
    fx, fy = (_array(state, "scratch:" + slot, output=True) for slot in ("diff6_x", "diff6_y"))
    w_launches = _smag_w_launches(state, common, dims, km, rw, fx, fy)
    flux = _raw(state, "smag2d", "wrf_smag_flux_s", _common_fields(state) +
                (("f", "thp0"), ("kh", "scratch:smag_kh"), ("thb", "thb"),
                 ("fx", "scratch:diff6_x"), ("fy", "scratch:diff6_y")),
                common + (_array(state, "thp0"), kh, _array(state, "thb"), np.int32(1),
                          np.int32(len(state.storage.specs["thb"].shape) == 3), fx, fy) + dims,
                ((cfg.nx + 1 + _TPB - 1) // _TPB, cfg.ny + 1, cfg.nz))
    divergence = _raw(state, "smag2d", "wrf_smag_hd_s", _common_fields(state) +
                      (("fx", "scratch:diff6_x"), ("fy", "scratch:diff6_y"), ("tend", "scratch:smag_rth")),
                      common + (fx, fy, rth) + dims, _grid(mass_shape))
    launches.extend(w_launches + (flux, divergence))
    vertical = []
    if cfg.bl_pbl_physics == 0:
        for axis, slot, field in (("u", "smag_ru", "u0"), ("v", "smag_rv", "v0"), ("w", "smag_rw", "w0")):
            value = _array(state, "scratch:" + slot, output=True)
            vertical.append(_raw(state, "smag2d", "wrf_smag_vd_" + axis, _common_fields(state) +
                                 (("km", "scratch:smag_km"), ("tend", "scratch:" + slot)),
                                 common + (km, value) + dims, _grid(state.storage.specs[field].shape)))
        for axis, slot, nxs, nys in (("u", "smag_ru", cfg.nx + 1, cfg.ny),
                                     ("v", "smag_rv", cfg.nx, cfg.ny + 1)):
            value = _array(state, "scratch:" + slot, output=True)
            if cfg.isfflx == 0:
                entry = "wrf_smag_surface_" + axis + "_cd0"
                fields = (("tend", "scratch:" + slot),)
                payload = (_F(cfg.tke_drag_coefficient), value)
            else:
                entry = "wrf_smag_surface_" + axis
                fields = (("ustm", "mup0"), ("tend", "scratch:" + slot))
                payload = (_array(state, "mup0"), np.int32(0), value)
            vertical.append(_raw(state, "smag2d", entry, _common_fields(state) + fields,
                                 common + payload + dims, ((nxs + _TPB - 1) // _TPB, nys, 1)))
        if cfg.isfflx in (0, 2):
            vertical.append(_raw(state, "smag2d", "wrf_smag_surface_heat_const", _common_fields(state) +
                                 (("hfx", "mup0"), ("rth", "scratch:smag_rth")),
                                 common + (_F(cfg.tke_heat_flux), _array(state, "mup0"), np.int32(0), rth) + dims,
                                 ((cfg.nx + _TPB - 1) // _TPB, cfg.ny, 1)))
        vertical.append(_raw(state, "smag2d", "wrf_smag_surface_scalars", _common_fields(state) +
                             (("hfx", "mup0"), ("qfx", "mup0"), ("rth", "scratch:smag_rth"), ("rqv", "scratch:smag_rth")),
                             common + (_array(state, "mup0"), _array(state, "mup0"), np.int32(0), np.int32(0), rth, rth) + dims,
                             ((cfg.nx + _TPB - 1) // _TPB, cfg.ny, 1)))
    launches.extend(vertical)
    ru, rv = (_array(state, "scratch:" + slot, output=True) for slot in ("smag_ru", "smag_rv"))

    def launch():
        for entry in first:
            entry()
        for temporary, entry in stresses:
            temporary.fill(0)
            entry()
            _zero_strips(temporary, cfg, 1)
        # All deformation tensors remain live through BOTH stress entries.
        cp.copyto(ru, stresses[0][0])
        cp.copyto(rv, stresses[1][0])
        rw.fill(0)
        for entry in w_launches:
            entry()
        _zero_strips(rw, cfg, 1)
        rth.fill(0)
        flux()
        divergence()
        _zero_strips(rth, cfg, 1)
        for entry in vertical:
            entry()
        for value in (ru, rv, rw, rth):
            _zero_strips(value, cfg, 1)

    launch.numerical_entries = tuple(entry for fn in launches for entry in fn.numerical_entries)
    return launch


def _diff6_launch(state, row, factor, mass):
    """One diff6 row, accumulated in place onto the held ``*_tendf`` slot.

    WRF forms ``(tendency + tendency_x) + tendency_y`` on the tendency it
    already holds (sixth_order_diffusion :6626), so the kernel writes the
    carrying slot itself; its loop bounds (``diff6_loop_bounds``) leave
    the points WRF skips untouched and the seam kernel adds the outermost
    staggered face.  ``row[2]`` (the former temporary slot) is unused.
    """
    from woof.core.dycore import _boundary_x, _boundary_y, _diff6_dt, diff6_loop_bounds
    field, slot, _temporary_slot, stagger, c1, c2 = row
    cfg = state.cfg
    name = "scratch:" + slot
    f = _array(state, field)
    nlev, nys, nxs = state.storage.specs[field].shape
    nx = nxs - 1 if stagger == "x" else nxs
    ny = nys - 1 if stagger == "y" else nys
    variant = 1 if stagger == "x" else (2 if stagger == "y" else 0)
    coef = _F(factor) * _F(0.015625) / (_F(2.0) * _F(_diff6_dt(cfg, slot)))
    slope = int(cfg.diff_6th_slopeopt) >= 1 and len(state.storage.specs["phb"].shape) == 3
    if slope and (cfg.dx <= 0.0 or cfg.dy <= 0.0):
        raise ValueError("diff_6th_slopeopt >= 1 needs positive dx/dy")
    phb = "phb" if slope else mass
    dzx = (_F(cfg.diff_6th_thresh) * _F(9.81) * (_F(1.0) / _F(1.0 / cfg.dx))) if slope else _F(0)
    dzy = (_F(cfg.diff_6th_thresh) * _F(9.81) * (_F(1.0) / _F(1.0 / cfg.dy))) if slope else _F(0)
    fields = (("f", field), ("tend", name), ("mut", mass), ("c1", c1), ("c2", c2),
              ("phb", phb), ("msfu", "msfu"), ("msfv", "msfv"), ("msft", "msft"))
    pointers = tuple(_array(state, allocation) for _, allocation in fields)
    scalars = (coef, np.int32(cfg.diff_6th_opt), np.int32(slope), dzx, dzy)
    seam_u = stagger == "x" and _boundary_x(cfg)
    seam_v = stagger == "y" and _boundary_y(cfg)
    bounds = diff6_loop_bounds(nx, ny, nxs, nys, _boundary_x(cfg), _boundary_y(cfg))
    main = _raw(state, "diff6", "diff6", fields, pointers + scalars + tuple(np.int32(value) for value in
                (nlev, ny, nys, nx, nxs, variant, stagger == "z", *bounds)), _grid((nlev, nys, nxs)))
    seam = None
    if seam_u or seam_v:
        along, cross = (nx, ny) if seam_u else (ny, nx)
        bnd_cross = _boundary_y(cfg) if seam_u else _boundary_x(cfg)
        h0, h1 = (3, cross - 4) if bnd_cross else (0, cross - 1)
        if along >= 6 and h1 >= h0:
            entry = "diff6_seam_u" if seam_u else "diff6_seam_v"
            seam = _raw(state, "diff6_seam", entry, fields, pointers + scalars + tuple(np.int32(value) for value in
                        (nlev, ny, nx, h0, h1, bnd_cross)), ((h1 - h0 + _TPB) // _TPB, 1, nlev))

    def launch():
        main()
        if seam is not None:
            seam()

    launch.numerical_entries = main.numerical_entries + (() if seam is None else seam.numerical_entries)
    return launch


def prepare_fixed_tendencies(state, *, mass="mixing_mut", inverse="mixing_inv_msfv"):
    """Bind the dry saved-field mixing sequence with no advancing member loop.

    All carriers must already be declared, admitted and copied. N=1 keeps
    original raw handles; the independent reference uses the untouched full
    scalar helper. Array ufunc outputs reuse the explicitly priced carriers.
    """
    import cupy as cp
    from woof.core.dycore import _clock_scaled_diff6_factor, diff6_exempt_slots
    cfg, bigstep = _context(state)
    if cfg.km_opt != 4 and cfg.diff_6th_opt <= 0:
        def no_work():
            return None
        no_work.numerical_entries = ()
        no_work.scalar_tendencies = {}
        return no_work
    held = tuple(_array(state, "scratch:" + row[1], output=True) for row in _ROWS)
    total_mass = _array(state, mass, output=True)
    if state.storage.specs[mass].shape != state.storage.specs["mup"].shape:
        raise ValueError("mixing total-mass carrier has a different column shape")
    base, perturbation = _array(state, "mub2d"), _array(state, "mup0")
    mixing = _smag_launches(state) if cfg.km_opt == 4 else None
    scalar_fixed = None
    if cfg.moist:
        from woof.ensemble.batch_moist_mixing import prepare_scalar_fixed_tendencies
        scalar_fixed = prepare_scalar_fixed_tendencies(state, mass=mass)
    diff6 = []
    if cfg.diff_6th_opt > 0:
        factor = _clock_scaled_diff6_factor(cfg)
        exempt = diff6_exempt_slots(cfg)
        diff6 = [_diff6_launch(state, row, factor, mass) for row in _ROWS if row[1] not in exempt]
    map_rows = []
    reciprocal = None
    reciprocal_input = None
    if state.has_msf:
        for row, value in zip(_ROWS, held):
            map_name = "msfu" if row[3] == "x" else ("msfv" if row[3] == "y" else "msft")
            factor = _array(state, map_name)
            # Insert the scalar level axis after the member axis, only for a
            # member-owned map. A shared map broadcasts without a new slab.
            map_view = factor[:, None] if state.storage.specs[map_name].ownership == "member" else factor[None]
            map_rows.append((value, map_view, bigstep and row[3] == "y"))
        if bigstep:
            reciprocal = _array(state, inverse, output=True)
            reciprocal_input = _array(state, "msfv_inv")
            if state.storage.specs[inverse].shape != state.storage.specs["msfv"].shape:
                raise ValueError("mixing v reciprocal carrier has a different map shape")
    device = int(cp.cuda.runtime.getDevice())
    bound = tuple((name, id(value)) for name, value in state.storage.arrays.items())
    configuration = _exact_key(cfg)

    def launch():
        if int(cp.cuda.runtime.getDevice()) != device:
            raise ValueError("prepared mixing belongs to another CUDA device")
        if bound != tuple((name, id(value)) for name, value in state.storage.arrays.items()):
            raise BatchStateUnsupported("mixing backings changed; rebind before submitting")
        if _exact_key(state.cfg) != configuration:
            raise BatchStateUnsupported("mixing configuration changed; rebind its scalar conversions and boundary calendar")
        for value in held:
            value.fill(0)
        if scalar_fixed is not None:
            scalar_fixed.clear()
        cp.add(base, perturbation, out=total_mass)
        if mixing is not None:
            mixing()
        if scalar_fixed is not None:
            scalar_fixed.horizontal()
        for entry in diff6:
            entry()
        if scalar_fixed is not None:
            scalar_fixed.diff6()
        if reciprocal is not None:
            cp.copyto(reciprocal, reciprocal_input)
        for value, map_view, multiply_reciprocal in map_rows:
            if multiply_reciprocal:
                cp.multiply(value, reciprocal[:, None], out=value)
            else:
                cp.divide(value, map_view, out=value)
        if scalar_fixed is not None:
            scalar_fixed.map_divide()

    launch.numerical_entries = (() if mixing is None else mixing.numerical_entries) + tuple(
        entry for fn in diff6 for entry in fn.numerical_entries) + (
            () if scalar_fixed is None else scalar_fixed.numerical_entries)
    launch.scalar_tendencies = {} if scalar_fixed is None else scalar_fixed.tendencies
    return launch


def prepare_add_fixed_dry_tendencies(state):
    """Bind the original held-tendency FP32 additions for a slow RK pass."""
    cfg, _ = _context(state)
    launches = []
    if cfg.km_opt == 4 or cfg.diff_6th_opt > 0:
        for row, target in zip(_ROWS, ("ru_t", "rv_t", "rw_t", "rth_t")):
            source = "scratch:" + row[1]
            size = int(np.prod(state.storage.specs[target].shape))
            launches.append(_raw(state, "bandwidth_glue", "glue_add",
                                 (("src", source), ("dst", target)),
                                 (_array(state, source), _array(state, target, output=True), np.uint64(size)),
                                 ((size + 511) // 512,)))

    def launch():
        for entry in launches:
            entry()

    launch.numerical_entries = tuple(entry for fn in launches for entry in fn.numerical_entries)
    return launch
