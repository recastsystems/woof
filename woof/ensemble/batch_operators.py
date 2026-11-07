"""Member-slab launch primitives for advection, diffusion and damping.

These adapters retain existing CUDA arithmetic in advection and diffusion. They
bind complete ``(member, level, row, column)`` allocations and shared vertical
profiles, then launch one numerical operation across the member dimension.
They do not initialize a forecast, couple tendencies, select a clock or attach
physics. Allocation and forecast admission remain the caller's responsibility.
"""

from __future__ import annotations

import math

import numpy as np
from woof.core.kernel_shape import shape_defines

from woof.grid_requirements import FIFTH_ORDER_STENCIL_AXIS
from woof.ensemble.batch_kernel import KernelSpec, PointerSpec, prepare_batch_kernel_launch

_TPB = 128
_FLOAT32 = np.dtype("float32")

ADD_DIFF2_SPEC = KernelSpec("diffusion", "add_diff2", (
    PointerSpec("f", "member"),
    PointerSpec("tend", "member"),
    PointerSpec("rdzf", "shared"),
    PointerSpec("rdzc", "shared"),
))
RAYLEIGH_DAMP_SPEC = KernelSpec("diffusion", "rayleigh_damp", (
    PointerSpec("f", "member"),
    PointerSpec("rdamp", "shared"),
))


def _flux_spec(entry, field, spacing):
    return KernelSpec("advection", entry, (
        PointerSpec(field, "member"),
        PointerSpec("ru", "member"),
        PointerSpec("rv", "member"),
        PointerSpec("rw", "member"),
        PointerSpec("tend_out", "member"),
        PointerSpec(spacing, "shared"),
        PointerSpec("fnm", "shared"),
        PointerSpec("fnp", "shared"),
        PointerSpec("msf", "shared"),
    ))


FLUX_DIV_SCALAR_SPEC = _flux_spec("flux_div_scalar", "q", "rdnw")
FLUX_DIV_U_SPEC = _flux_spec("flux_div_u", "u", "rdnw")
FLUX_DIV_V_SPEC = _flux_spec("flux_div_v", "v", "rdnw")
FLUX_DIV_W_SPEC = _flux_spec("flux_div_w", "w", "rdn")
_FLUX_SPECS = {"": FLUX_DIV_SCALAR_SPEC, "x": FLUX_DIV_U_SPEC,
               "y": FLUX_DIV_V_SPEC, "z": FLUX_DIV_W_SPEC}


def fixed_step_seconds(dt) -> float:
    """Validate one explicit common model step without choosing a minimum.

    This scalar contract is suitable for configuring same-step identity
    experiments. It does not qualify an adaptive forecast as fixed-step.
    """
    if isinstance(dt, (bool, np.bool_)) or np.ndim(dt) != 0:
        raise TypeError("fixed model step must be one real scalar in seconds")
    value = float(dt)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("fixed model step must be finite and positive")
    return value


def _cuda_float32(array, name):
    if not hasattr(array, "__cuda_array_interface__"):
        raise TypeError(f"{name} needs a CUDA array backing")
    if array.dtype != _FLOAT32 or not array.flags.c_contiguous:
        raise ValueError(f"{name} needs contiguous float32 storage")
    if not array.nbytes:
        raise ValueError(f"{name} needs nonempty storage")
    return array


def _member_field(array, name):
    _cuda_float32(array, name)
    if array.ndim != 4 or any(extent < 1 for extent in array.shape):
        raise ValueError(
            f"{name} must expose the complete nonempty "
            "(members, levels, rows, columns) backing")
    members, nlev, nys, nxs = map(int, array.shape)
    stride = int(array.strides[0])
    if stride != nlev * nys * nxs * _FLOAT32.itemsize:
        raise ValueError(f"{name} member byte stride differs from its spatial slab")
    return members, nlev, nys, nxs, stride


def _shared_profile(array, name, length, *, dummy_when_empty=False):
    _cuda_float32(array, name)
    expected = max(1, length) if dummy_when_empty else length
    if array.shape != (expected,):
        detail = " (one unused element for a one-level field)" if length == 0 else ""
        raise ValueError(f"{name} must have shape ({expected},){detail}")


def _interval(array):
    interface = array.__cuda_array_interface__
    start = int(interface["data"][0])
    return start, start + int(array.nbytes)


def _separate_output(output, inputs):
    start, end = _interval(output)
    for name, array in inputs:
        other_start, other_end = _interval(array)
        if start < other_end and other_start < end:
            raise ValueError(
                f"output overlaps {name}; concurrent stencil writes would "
                "change member inputs or shared vertical coefficients")


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise TypeError(f"{name} must be one real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _kernel_scalar(value, name):
    with np.errstate(over="ignore"):
        converted = np.float32(_finite(value, name))
    if not np.isfinite(converted):
        raise ValueError(f"{name} is outside the float32 kernel argument range")
    return converted


def prepare_add_diff2(f, tend, rdzf, rdzc, *, kh, kv, dx, dy,
                      stagger=""):
    """Bind one uncoupled constant-K diffusion operation for every member.

    ``f`` and ``tend`` are complete contiguous four-dimensional float32 CUDA
    arrays with equal shapes. ``stagger`` is the scalar launcher's mass, x,
    y or z spelling. ``rdzf`` and ``rdzc`` are precomputed shared float32
    profiles, with lengths ``nlev-1`` and ``nlev``. A one-level field supplies
    one unused ``rdzf`` element so its CUDA pointer has nonempty backing.

    The closure adds to the existing tendencies. It allocates no device arrays
    and retains its inputs until all launches have finished. N=1 uses the
    current scalar loader with the same base addresses and scalar arguments.
    """
    members, nlev, nys, nxs, field_stride = _member_field(f, "f")
    tend_shape = _member_field(tend, "tend")
    if tend_shape[:4] != (members, nlev, nys, nxs):
        raise ValueError("diffusion field and tendency member/spatial shapes differ")
    if stagger not in ("", "x", "y", "z"):
        raise ValueError("diffusion staggering must be '', 'x', 'y' or 'z'")
    nx = nxs - int(stagger == "x")
    ny = nys - int(stagger == "y")
    if nx < 1 or ny < 1:
        raise ValueError("diffusion staggering leaves no periodic spatial core")
    _shared_profile(rdzf, "rdzf", nlev - 1, dummy_when_empty=True)
    _shared_profile(rdzc, "rdzc", nlev)
    _separate_output(tend, (("f", f), ("rdzf", rdzf), ("rdzc", rdzc)))
    kh, kv = _kernel_scalar(kh, "kh"), _kernel_scalar(kv, "kv")
    dx, dy = _finite(dx, "dx"), _finite(dy, "dy")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("diffusion dx and dy must be positive to define inverse spacing")
    try:
        dx_inv2, dy_inv2 = 1.0 / dx ** 2, 1.0 / dy ** 2
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError("diffusion spacing cannot define finite inverse squares") from error
    dx_inv2 = _kernel_scalar(dx_inv2, "inverse squared dx")
    dy_inv2 = _kernel_scalar(dy_inv2, "inverse squared dy")
    args = (f, tend, kh, kv, dx_inv2, dy_inv2, rdzf, rdzc,
            np.int32(nlev), np.int32(ny), np.int32(nys),
            np.int32(nx), np.int32(nxs), np.int32(stagger == "z"))
    strides = {"f": field_stride, "tend": tend_shape[4],
               "rdzf": 0, "rdzc": 0}
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    return prepare_batch_kernel_launch(ADD_DIFF2_SPEC, members, grid, (_TPB, 1, 1),
                                      args, pointer_strides=strides)


def prepare_rayleigh_damp(f, rdamp):
    """Bind the existing in-place per-level damping arithmetic for all members.

    ``rdamp`` is one shared, precomputed profile with ``nlev`` float32
    elements. This is the existing direct utility, not the acoustic w damper
    or a forecast driver. Passing each spatial staggering preserves the
    scalar kernel's behavior for that field; no field selection is implied.
    """
    members, nlev, nys, nxs, stride = _member_field(f, "f")
    _shared_profile(rdamp, "rdamp", nlev)
    _separate_output(f, (("rdamp", rdamp),))
    plane = nys * nxs
    args = (f, rdamp, np.int32(nlev), np.int32(plane))
    grid = ((nlev * plane + _TPB - 1) // _TPB, 1, 1)
    strides = {"f": stride, "rdamp": 0}
    return prepare_batch_kernel_launch(RAYLEIGH_DAMP_SPEC, members, grid, (_TPB, 1, 1),
                                      args, pointer_strides=strides)


def advection_order_defines(horder=5, *, theta_offset=False, stagger=""):
    """The integer tier the single-member launcher compiles for ``horder``.

    Mirrors ``woof.core.advection._launch``: order 5 is the untiered module,
    order 0 is ``GPUWM_NO_HADV`` (plus ``GPUWM_THETA_OFFSET`` for the theta
    row when the scalar driver asks for it), and 2/3/4/6 are ``GPUWM_HORDER``.
    A batch launch compiled at another tier differs from a single run.
    """
    if isinstance(horder, (bool, np.bool_)) or int(horder) not in (0, 2, 3, 4, 5, 6):
        raise ValueError("advection horder must be 0, 2, 3, 4, 5 or 6")
    horder = int(horder)
    if theta_offset and (horder != 0 or stagger != ""):
        raise ValueError("the theta offset tier exists only for the scalar row with horder 0")
    if horder == 5:
        return ()
    defines = (("GPUWM_NO_HADV", 1),) if horder == 0 else (("GPUWM_HORDER", horder),)
    if theta_offset:
        defines += (("GPUWM_THETA_OFFSET", 1),)
    return defines


def flux_row_horizontal_orders(cfg):
    """Per-staggering horizontal orders exactly as the scalar dycore passes them."""
    from woof.core.advection import scalar_horizontal_order
    scalar = scalar_horizontal_order(cfg)
    momentum = int(getattr(cfg, "h_mom_adv_order", 5))
    return {"": scalar, "x": momentum, "y": momentum, "z": scalar}


def prepare_flux_div(field, ru, rv, rw, tend, spacing, fnm, fnp, msf, *,
                     dx, dy, stagger="", open_x=False, open_y=False,
                     has_msf=False, spec=False, vorder=3, horder=5,
                     theta_offset=False):
    """Bind existing flux-form advection arithmetic over a member batch.

    Field, mass-flux and tendency arrays expose complete four-dimensional
    member backings, with their original C-grid staggerings. ``spacing`` is
    shared ``rdnw`` for scalar/u/v and shared ``rdn`` for w; ``fnm``/``fnp``
    are shared vertical interpolation weights. ``msf`` is the shared 2-D map
    factor at the target staggering, including when ``has_msf`` is false.
    The caller prepares these arrays and chooses the same boundary/map flags
    as its single-member run. The adapter allocates no CUDA field or profile.

    This primitive adds to ``tend``. It neither constructs the mass fluxes
    nor couples the resulting tendencies or advances RK/acoustic state.
    ``vorder`` is WRF's vertical order for this field (3 or 5), the same
    value the single-member launcher takes; ``horder``/``theta_offset`` are
    its horizontal order tier (see :func:`advection_order_defines`).
    """
    members, nlev, nys, nxs, stride = _member_field(field, "field")
    order_defines = advection_order_defines(horder, theta_offset=theta_offset, stagger=stagger)
    if vorder not in (3, 5):
        raise ValueError("advection vorder must be 3 or 5 (the WRF vert_order ladders the kernel carries)")
    if stagger not in _FLUX_SPECS:
        raise ValueError("advection staggering must be '', 'x', 'y' or 'z'")
    nz = nlev - int(stagger == "z")
    ny = nys - int(stagger == "y")
    nx = nxs - int(stagger == "x")
    if min(nz, ny, nx) < 1:
        raise ValueError("advection staggering leaves no mass-point spatial core")
    shapes = {"ru": (members, nz, ny, nx + 1),
              "rv": (members, nz, ny + 1, nx),
              "rw": (members, nz + 1, ny, nx),
              "tend_out": tuple(field.shape)}
    strides = {}
    inputs = (("ru", ru), ("rv", rv), ("rw", rw))
    for name, array in inputs + (("tend_out", tend),):
        dims = _member_field(array, name)
        if tuple(array.shape) != shapes[name]:
            raise ValueError(f"{name} must have member/C-grid shape {shapes[name]}")
        strides[name] = dims[4]
    if open_x and nx < FIFTH_ORDER_STENCIL_AXIS:
        raise ValueError("open_x advection needs nx >= 7 so boundary degrade bands do not overlap")
    if open_y and ny < FIFTH_ORDER_STENCIL_AXIS:
        raise ValueError("open_y advection needs ny >= 7 so boundary degrade bands do not overlap")
    kernel_spec = _FLUX_SPECS[stagger]
    field_name, spacing_name = kernel_spec.pointers[0].name, kernel_spec.pointers[5].name
    for name, profile in ((spacing_name, spacing), ("fnm", fnm), ("fnp", fnp)):
        _shared_profile(profile, name, nz)
    _cuda_float32(msf, "msf")
    if msf.shape != (nys, nxs):
        raise ValueError(f"msf must have target-grid shape {(nys, nxs)}")
    _separate_output(tend, ((field_name, field),) + inputs + (
        (spacing_name, spacing), ("fnm", fnm), ("fnp", fnp), ("msf", msf)))
    dx, dy = _finite(dx, "dx"), _finite(dy, "dy")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("advection dx and dy must be positive to define inverse spacing")
    dx_inv, dy_inv = _kernel_scalar(1.0 / dx, "inverse dx"), _kernel_scalar(1.0 / dy, "inverse dy")
    strides.update({field_name: stride, spacing_name: 0, "fnm": 0, "fnp": 0, "msf": 0})
    args = (field, ru, rv, rw, tend, spacing, fnm, fnp, msf, dx_inv, dy_inv,
            np.int32(nz), np.int32(ny), np.int32(nx),
            np.int32(open_x), np.int32(open_y), np.int32(has_msf), np.int32(spec),
            np.int32(vorder))
    grid = ((nxs + _TPB - 1) // _TPB, nys, nlev)
    return prepare_batch_kernel_launch(kernel_spec, members, grid, (_TPB, 1, 1),
                                      args, pointer_strides=strides,
                                      defines=order_defines + shape_defines(
                                          kernel_spec.module, kernel_spec.entry, (nz, ny, nx)))


def prepare_flux_div_scalar(q, ru, rv, rw, tend, rdnw, fnm, fnp, msf, **options):
    """Bind the existing mass-point scalar flux-divergence entry."""
    return prepare_flux_div(q, ru, rv, rw, tend, rdnw, fnm, fnp, msf,
                            stagger="", **options)
