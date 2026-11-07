"""Inventoried member-wide array operations used between dycore launches.

Outputs are allocated during admission, never during submission. Existing raw
sources and CuPy FP32 operations retain the scalar rounding boundaries. This
module does not choose a timestep, prepare inputs, or integrate a forecast.
"""
from __future__ import annotations

import numpy as np

from woof.ensemble.batch_kernel import KernelSpec, PointerSpec, prepare_batch_kernel_launch
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported
from woof.ensemble.batch_storage import BatchArraySpec


def workspace_specs(cfg):
    """The complete extra member carriers used by these prepared operations."""
    return (
        BatchArraySpec("batch_mass", (cfg.ny, cfg.nx), "member"),
        BatchArraySpec("batch_mux", (cfg.ny, cfg.nx + 1), "member"),
        BatchArraySpec("batch_muy", (cfg.ny + 1, cfg.nx), "member"),
        BatchArraySpec("batch_theta", (cfg.nz, cfg.ny, cfg.nx), "member"),
    )


def _array(state, name, shape, *, output=False):
    if not isinstance(state, BatchedDomainState):
        raise TypeError("prepared member glue needs an admitted BatchedDomainState")
    spec = state.storage.specs.get(name)
    if spec is None:
        raise BatchStateUnsupported(f"glue carrier {name!r} is not in the admission inventory")
    value = state.storage.arrays[name]
    if (spec.shape != tuple(shape) or np.dtype(spec.dtype) != np.dtype(np.float32)
            or value.dtype != np.dtype(np.float32)
            or tuple(value.shape) != spec.allocation_shape(state.members)
            or not value.flags.c_contiguous
            or not hasattr(value, "__cuda_array_interface__")):
        raise ValueError(f"glue carrier {name!r} differs from its admitted shape/dtype")
    if output and spec.ownership != "member":
        raise BatchStateUnsupported(f"glue output {name!r} must be independent for each member")
    return value


def _bind(state, module, entry, bindings, args, grid, threads=128):
    specs = tuple(PointerSpec(parameter, state.storage.specs[name].ownership)
                  for parameter, name in bindings)
    strides = {parameter: state.storage.pointer_stride_bytes(name)
               for parameter, name in bindings}
    return prepare_batch_kernel_launch(
        KernelSpec(module, entry, specs), state.members, grid, (threads,), args,
        pointer_strides=strides)


def _disjoint(output, inputs):
    lo, hi = output.data.ptr, output.data.ptr + output.nbytes
    for value in inputs:
        if lo < value.data.ptr + value.nbytes and value.data.ptr < hi:
            raise ValueError("glue output overlaps an input and changes its scalar operand")


def prepare_total_mass(state, *, out="batch_mass"):
    """Bind the scalar mub2d + mup FP32 addition across independent members."""
    import cupy as cp
    shape = (state.cfg.ny, state.cfg.nx)
    base = _array(state, "mub2d", shape)
    perturbation = _array(state, "mup", shape)
    result = _array(state, out, shape, output=True)
    _disjoint(result, (base, perturbation))

    def launch():
        cp.add(base, perturbation, out=result)

    return launch


def prepare_total_theta(state, *, out="batch_theta", perturbation="thp"):
    """Bind the existing glue_total_theta body into a priced output backing."""
    cfg = state.cfg
    shape = (cfg.nz, cfg.ny, cfg.nx)
    base_shape = state.storage.specs["thb"].shape
    if base_shape not in ((cfg.nz,), shape):
        raise ValueError("theta base has neither original scalar layout")
    base = _array(state, "thb", base_shape)
    perturbation_array = _array(state, perturbation, shape)
    result = _array(state, out, shape, output=True)
    _disjoint(result, (base, perturbation_array))
    size = cfg.nz * cfg.ny * cfg.nx
    return _bind(state, "bandwidth_glue", "glue_total_theta",
                 (("thb", "thb"), ("thp", perturbation), ("dst", out)),
                 (base, perturbation_array, result, np.uint64(size),
                  np.int32(cfg.ny * cfg.nx), np.int32(len(base_shape) == 3)),
                 ((size + 511) // 512,))


def prepare_face_masses(state, *, mass="batch_mass", mux="batch_mux", muy="batch_muy"):
    """Preserve each stage_face_masses branch in explicitly priced outputs."""
    import cupy as cp
    from woof.core.dycore import _boundary_x, _boundary_y
    from woof.wrf_exact import ENABLED
    if ENABLED:
        return _prepare_strict_face_masses(state, mux=mux, muy=muy)
    cfg = state.cfg
    source = _array(state, mass, (cfg.ny, cfg.nx), output=True)
    outputs = (_array(state, mux, (cfg.ny, cfg.nx + 1), output=True),
               _array(state, muy, (cfg.ny + 1, cfg.nx), output=True))
    launches = []
    for axis, name, target in ((0, mux, outputs[0]), (1, muy, outputs[1])):
        _disjoint(target, (source,))
        size = int(np.prod(state.storage.specs[name].shape))
        launches.append(_bind(
            state, "face_mass", "average_mass_faces", (("mu", mass), ("out", name)),
            (source, target, np.int32(cfg.ny), np.int32(cfg.nx), np.int32(axis)),
            ((size + 127) // 128,)))
    _disjoint(outputs[0], (outputs[1],))
    x_boundary, y_boundary = _boundary_x(cfg), _boundary_y(cfg)

    def launch():
        launches[0]()
        launches[1]()
        if x_boundary:
            cp.copyto(outputs[0][..., 0], source[..., 0])
            cp.copyto(outputs[0][..., -1], source[..., -1])
        if y_boundary:
            cp.copyto(outputs[1][:, 0, :], source[:, 0, :])
            cp.copyto(outputs[1][:, -1, :], source[:, -1, :])

    return launch


def _prepare_strict_face_masses(state, *, mux, muy):
    """Keep calc_mu_uv's four FP32 round points without roll/concatenate.

    The independent scalar helper adds both perturbations, then the current
    base word, then the neighbour base word, then multiplies by one half.
    Each member-wide CuPy operation writes its next round point directly
    into an admitted output slice. Periodic aliases and forced boundaries
    retain the scalar branch's operand order, including N=1.
    """
    import cupy as cp
    from woof.core.ieva import _periodic_x, _periodic_y
    cfg = state.cfg
    shape = (cfg.ny, cfg.nx)
    perturbation = _array(state, "mup", shape)
    base = _array(state, "mub2d", shape)
    outputs = (_array(state, mux, (cfg.ny, cfg.nx + 1), output=True),
               _array(state, muy, (cfg.ny + 1, cfg.nx), output=True))
    for result in outputs:
        _disjoint(result, (perturbation, base))
    _disjoint(outputs[0], (outputs[1],))
    periodic_x, periodic_y = _periodic_x(cfg), _periodic_y(cfg)
    half = np.float32(0.5)
    close_faces = _prepare_close_faces(
        state, x=mux, y=muy, close_x=periodic_x, close_y=periodic_y)

    def average(out, m, neighbour_m, b, neighbour_b):
        cp.add(m, neighbour_m, out=out)
        cp.add(out, b, out=out)
        cp.add(out, neighbour_b, out=out)
        cp.multiply(half, out, out=out)

    def launch():
        x, y = outputs
        average(x[..., 1:cfg.nx], perturbation[..., 1:], perturbation[..., :-1],
                base[..., 1:], base[..., :-1])
        average(y[:, 1:cfg.ny, :], perturbation[:, 1:, :], perturbation[:, :-1, :],
                base[..., 1:, :], base[..., :-1, :])
        if periodic_x:
            average(x[..., 0], perturbation[..., 0], perturbation[..., -1],
                    base[..., 0], base[..., -1])
        else:
            for face in (0, -1):
                m, b = perturbation[..., face], base[..., face]
                average(x[..., face], m, m, b, b)
        if periodic_y:
            average(y[:, 0, :], perturbation[:, 0, :], perturbation[:, -1, :],
                    base[..., 0, :], base[..., -1, :])
        else:
            for face in (0, -1):
                m, b = perturbation[:, face, :], base[..., face, :]
                average(y[:, face, :], m, m, b, b)
        close_faces()

    return launch


def _prepare_close_faces(state, *, x, y, close_x, close_y, levels=None):
    """Close integer words of 2-D carriers or 3-D prognostics in one launch.

    A 2-D carrier has one logical level. ``levels`` supplies the vertical
    extent for a 3-D field, including a one-level prognostic. The source faces
    and destination faces are disjoint; outputs must be separate allocations.
    """
    import cupy as cp
    cfg = state.cfg
    prefix = () if levels is None else (levels,)
    nz = 1 if levels is None else levels
    u = _array(state, x, prefix + (cfg.ny, cfg.nx + 1), output=True)
    v = _array(state, y, prefix + (cfg.ny + 1, cfg.nx), output=True)
    _disjoint(u, (v,))
    if not close_x and not close_y:
        def unchanged():
            return None
        return unchanged
    n_u, n_v = nz * cfg.ny, nz * cfg.nx
    spec = KernelSpec("ensemble_bookkeeping", "ensemble_close_faces",
                      (PointerSpec("u", "member", "uint32"), PointerSpec("v", "member", "uint32")))
    return prepare_batch_kernel_launch(
        spec, state.members, ((max(n_u, n_v) + 127) // 128,), (128,),
        (u.view(cp.uint32), v.view(cp.uint32), np.uint64(n_u), np.uint64(n_v),
         np.int32(cfg.nx), np.int32(cfg.ny), np.int32(close_x), np.int32(close_y)),
        pointer_strides={"u": state.storage.pointer_stride_bytes(x),
                         "v": state.storage.pointer_stride_bytes(y)})


def prepare_periodic_alias(state):
    """Bind the scalar epilogue's exact staggered face copies across members."""
    from woof.core.dycore import _boundary_x, _boundary_y
    cfg = state.cfg
    return _prepare_close_faces(
        state, x="u", y="v", levels=cfg.nz,
        close_x=not _boundary_x(cfg), close_y=not _boundary_y(cfg))
