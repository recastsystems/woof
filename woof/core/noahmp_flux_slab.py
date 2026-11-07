"""Whole-slab CuPy packing for the Noah-MP VEGE_FLUX and BARE_FLUX leaves.

:mod:`woof.core.noahmp_vegeflux_gpu` and :mod:`woof.core.noahmp_bareflux_gpu`
batch a *list of Python call tuples*: one dict per land column, and one CPython
loop iteration per column to copy 50 (VEGE_FLUX) or 53 (BARE_FLUX) values into
a flat row.  That loop -- not the CUDA launch -- is what makes Noah-MP cost
wall hours per simulated minute at d04.

This module packs the identical rows out of whole-slab CuPy arrays instead.
Everything downstream of the packing is the *existing* code reached through the
existing entry points: the same ``noahmp_vegeflux.cu`` / ``noahmp_bareflux.cu``
kernels, the same ``noahmp_vegeflux_gpu._module`` (so the same
``__constant__`` table uploads, done once by its ``lru_cache``), the same
``woof.core.kernels.get_kernel``, the same block size, the same slot order and
the same dtypes.  There is no arithmetic in this file at all: the only change
is that ``inputs[row] = [...]`` over n columns becomes
``inputs[:, slot] = fields[name]`` over ~50 slots.

Three things are essential and are why this is not a two-line rewrite:

* **The slot order is imported, never re-spelled.**  Every loop below iterates
  the owning module's own ``INPUT_NAMES`` / ``PARAMETER_NAMES`` /
  ``SCALAR_NAMES`` / ``ARRAY_NAMES`` tuple.  A second, hand-written copy of a
  50-slot order is a silent forecast change waiting for someone to edit one of
  them.
* **The layer axis is last, in Fortran order.**  The per-column packer writes
  ``slot = layer + (NSNOW - 1)`` for ``layer`` in ``-nsnow+1 .. nsoil``, i.e.
  the dicts keyed by Fortran layer index become columns ``0 .. NLAYER-1``.  The
  slab form takes those as ``(n, NLAYER)`` in exactly that order.
* **Everything handed to a raw kernel is contiguous.**  A raw-kernel argument
  carries a bare pointer and no stride, so a CuPy *view* silently reads the
  wrong memory -- a trap this project has already been bitten by twice.  Every
  array that reaches a launch here goes through ``cupy.ascontiguousarray``,
  including the ones a caller supplies (``dzsnso``/``stc``/``df``/``isnow``
  are passed straight through to ``k_vegeflux``), and the returned output
  columns are rows of one contiguous transpose rather than strided slices of
  the output slab, so a caller may forward them to another kernel.

BARE_FLUX runs on every land column; VEGE_FLUX only on vegetated tiles.  These
functions are general over whatever subset of rows they are handed -- the
caller does the selection.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

from woof.core.noahmp_slab_libm import copy_slab_slots

import numpy as np

from woof.core import noahmp_bareflux_gpu as _bare
from woof.core import noahmp_vegeflux_gpu as _vege

NSNOW = _vege.NSNOW
NSOIL = _vege.NSOIL
NLAYER = _vege.NLAYER

if (_bare.NSNOW, _bare.NSOIL, _bare.NLAYER) != (NSNOW, NSOIL, NLAYER):
    raise ImportError(
        "VEGE_FLUX and BARE_FLUX disagree about the layer geometry: "
        f"{(_vege.NSNOW, _vege.NSOIL, _vege.NLAYER)} vs "
        f"{(_bare.NSNOW, _bare.NSOIL, _bare.NLAYER)}")

#: Fortran layer index of packed slot ``0``: ``slot = layer + LAYER_OFFSET``.
LAYER_OFFSET = _vege.LAYER_OFFSET

#: Options with one implemented device branch. STC has its own 1/2/3 selector;
#: full-column admission still requires the jointly selected physics replay.
_VEGE_OPTIONS = (("opt_crop", 0),)
_BARE_OPTIONS = ()


class VegeFluxSlabInputs(NamedTuple):
    """The six device arguments ``k_vegeflux`` reads, in launch order.

    A plain tuple, so ``inputs, params, *rest = pack_vege_flux_slab(...)``
    works; the launch needs all six and hiding four of them behind the two the
    caller usually looks at would leave them untested.
    """

    inputs: object      # (n, 50) float32
    params: object      # (n, 14) float32
    isnow: object       # (n,)    int32
    dzsnso: object      # (n, 7)  float32
    stc: object         # (n, 7)  float32
    df: object          # (n, 7)  float32


# --------------------------------------------------------------------------
# field access
# --------------------------------------------------------------------------
def _cupy():
    import cupy as cp

    return cp


def _field(fields: Mapping[str, object], name: str, leaf: str):
    try:
        return fields[name]
    except KeyError:
        raise TypeError(f"{leaf} slab is missing field {name!r}") from None


def _column(cp, fields: Mapping[str, object], name: str, n: int, leaf: str):
    """One ``(n,)`` (or broadcastable 0-d) column, unconverted.

    Left in whatever dtype the caller supplied: the cast happens once, at the
    slice assignment into the float32 slab, exactly as the per-column packer's
    ``inputs[row] = [...]`` casts once on assignment.
    """
    arr = cp.asarray(_field(fields, name, leaf))
    if arr.ndim == 0:
        return arr
    if arr.shape != (n,):
        raise ValueError(
            f"{leaf} slab field {name!r} has shape {tuple(arr.shape)}; "
            f"expected ({n},) or a scalar")
    return arr


def _layers(cp, fields: Mapping[str, object], name: str, n: int, leaf: str):
    """One ``(n, NLAYER)`` float32 block, contiguous.

    The layer axis is last and runs ``-NSNOW+1 .. NSOIL`` -- the same order the
    per-column packer produces from the Fortran-indexed dicts.
    """
    arr = cp.asarray(_field(fields, name, leaf))
    if arr.shape != (n, NLAYER):
        raise ValueError(
            f"{leaf} slab field {name!r} has shape {tuple(arr.shape)}; "
            f"expected ({n}, {NLAYER}) with the layer axis last, "
            f"spanning {-NSNOW + 1}..{NSOIL}")
    return cp.ascontiguousarray(arr, dtype=cp.float32)


def _int_column(cp, fields: Mapping[str, object], name: str, n: int,
                leaf: str):
    """One contiguous ``(n,) int32`` column, broadcast if given as a scalar."""
    arr = _column(cp, fields, name, n, leaf)
    if arr.ndim == 0:
        arr = cp.broadcast_to(arr, (n,))
    return cp.ascontiguousarray(arr, dtype=cp.int32)


def _check_size(n: int) -> int:
    n = int(n)
    if n < 0:
        raise ValueError(f"a slab cannot have {n} columns")
    return n


def _check_options(fields: Mapping[str, object], leaf: str,
                   admitted: Sequence[tuple[str, int]]) -> None:
    """Refuse the options the device leaves do not transcribe.

    The options are a compile-time identity for the whole domain, never a
    per-column value, so an array here is a mistake worth naming rather than
    reducing on the device (which would cost a synchronisation per step).
    """
    for name, value in (("nsnow", NSNOW), ("nsoil", NSOIL)) + tuple(admitted):
        if name not in fields:
            continue
        got = fields[name]
        if getattr(got, "ndim", 0):
            raise TypeError(
                f"device {leaf} takes {name} as a scalar, not a per-column "
                "array")
        if int(got) != value:
            if name in ("nsnow", "nsoil"):
                raise ValueError(
                    f"the runtime CUDA {leaf} is pinned to {NSNOW} snow and "
                    f"{NSOIL} soil layers, got {name}={int(got)}")
            raise NotImplementedError(
                f"device {leaf} admits {name}={value} only")


def _sfc_option(fields):
    value=fields.get("opt_sfc",1)
    if type(value) is not int or value not in (1,2):
        raise ValueError("flux opt_sfc must be scalar int 1 or 2")
    return value


def _crs_option(fields):
    value=fields.get("opt_crs",1)
    if type(value) is not int or value not in (1,2):
        raise ValueError("VEGE_FLUX opt_crs must be scalar int 1 or 2")
    return value


def _stc_option(fields: Mapping[str, object], leaf: str) -> int:
    """Select the implemented whole-domain STC branch without device readback."""
    value = fields.get("opt_stc", 1)
    if type(value) is not int:
        raise TypeError(f"device {leaf} opt_stc must be a scalar int")
    if value not in (1, 2, 3):
        raise NotImplementedError(f"device {leaf} admits opt_stc=1, 2 or 3 only")
    return value


def _unpack(cp, device_out, names: Sequence[str]) -> dict:
    """Split an ``(n, len(names))`` output slab into named ``(n,)`` columns.

    One transpose, not ``len(names)`` copies -- and every returned column is a
    row of a C-contiguous array, so it is safe to hand to another raw kernel.
    """
    columns = cp.ascontiguousarray(device_out.T)
    return {name: columns[index] for index, name in enumerate(names)}


# --------------------------------------------------------------------------
# VEGE_FLUX
# --------------------------------------------------------------------------
def pack_vege_flux_slab(fields: Mapping[str, object],
                        n: int) -> VegeFluxSlabInputs:
    """Pack ``n`` vegetated columns into ``k_vegeflux``'s six arguments.

    ``fields`` maps the argument names
    :func:`woof.core.noahmp_vegeflux_gpu._bind_call` binds to CuPy arrays:
    ``(n,)`` for every scalar, ``(n, NLAYER)`` for ``dzsnso``/``stc``/``df``,
    ``(n,)`` int for ``isnow``, and ``(n,)`` for each name in
    ``PARAMETER_NAMES`` -- i.e. the values the per-column path reads off the
    parameter object. Names VEGE_FLUX accepts but never packs (``latheag``, ``q2``, ``qc``, ``p``) are ignored if present. SFC2 packs actual THAIR/CZIL. STC=3 takes
    actual ``fsno`` separately; STC=1 does not read it.
    """
    _slots_inputs = []
    _slots_params = []
    cp = _cupy()
    n = _check_size(n)
    _check_options(fields, "VEGE_FLUX", _VEGE_OPTIONS)
    _stc_option(fields, "VEGE_FLUX")

    opt_sfc = _sfc_option(fields)
    inputs = cp.empty((n, _vege.N_INPUT + (opt_sfc == 2)), dtype=cp.float32)
    for slot, name in enumerate(_vege.INPUT_NAMES):
        _slots_inputs.append((slot, _column(cp, fields, name, n, "VEGE_FLUX")))

    if opt_sfc == 2:
        _slots_inputs.append((_vege.N_INPUT, _column(cp, fields, "thair", n, "VEGE_FLUX SFC2")))
    names=(_vege.PARAMETER_NAMES
           + (("RGL","RSMIN","RSMAX","TOPT","HS") if _crs_option(fields)==2 else ())
           + (("CZIL",) if opt_sfc==2 else ()))
    params = cp.empty((n, len(names)), dtype=cp.float32)
    for slot, name in enumerate(names):
        _slots_params.append((slot, _column(cp, fields, name, n, "VEGE_FLUX")))

    copy_slab_slots(params, _slots_params)
    copy_slab_slots(inputs, _slots_inputs)
    return VegeFluxSlabInputs(
        inputs=inputs,
        params=params,
        isnow=_int_column(cp, fields, "isnow", n, "VEGE_FLUX"),
        dzsnso=_layers(cp, fields, "dzsnso", n, "VEGE_FLUX"),
        stc=_layers(cp, fields, "stc", n, "VEGE_FLUX"),
        df=_layers(cp, fields, "df", n, "VEGE_FLUX"),
    )


def evaluate_vege_flux_slab(fields: Mapping[str, object], n: int, *,
                            use_device_libm: bool = False,
                            atan_profile="glibc_2_39") -> dict:
    """Run VEGE_FLUX over a whole slab; return ``OUTPUT_NAMES -> (n,)``.

    ``use_device_libm`` is the existing negative control and is forwarded
    unchanged; it is not a runtime option.
    """
    cp = _cupy()
    n = _check_size(n)
    if n == 0:
        return {name: cp.empty(0, dtype=cp.float32)
                for name in _vege.OUTPUT_NAMES}

    packed = pack_vege_flux_slab(fields, n)
    device_out = cp.empty((n, _vege.N_OUTPUT), dtype=cp.float32)
    opt_stc = _stc_option(fields, "VEGE_FLUX")
    kernel_name = f"k_vegeflux_stc{opt_stc}" if opt_stc != 1 else "k_vegeflux"
    opt_crs = _crs_option(fields)
    opt_sfc = _sfc_option(fields)
    if opt_crs == 2: kernel_name = "k_vegeflux_crs2"
    if opt_sfc == 2: kernel_name = "k_vegeflux_sfc2"
    profile_options = ({} if atan_profile == "glibc_2_39" else
                       {"atan_profile": atan_profile})
    module = (_vege._module(use_device_libm, selected_options=True, **profile_options)
              if opt_stc != 1 or opt_crs != 1 or opt_sfc != 1 else _vege._module(use_device_libm, **profile_options))
    kernel = module.get_function(kernel_name)
    blocks = (n + _vege.THREADS - 1) // _vege.THREADS
    args = (cp.ascontiguousarray(packed.inputs),
            cp.ascontiguousarray(packed.params),
            cp.ascontiguousarray(packed.isnow),
            cp.ascontiguousarray(packed.dzsnso),
            cp.ascontiguousarray(packed.stc),
            cp.ascontiguousarray(packed.df))
    if opt_stc == 3 or opt_crs == 2 or opt_sfc == 2:
        fsno = _column(cp, fields, "fsno", n, "VEGE_FLUX selected")
        args += (cp.ascontiguousarray(cp.broadcast_to(fsno, (n,)),
                                     dtype=cp.float32),)
    kernel(
        (blocks,), (_vege.THREADS,),
        args + (device_out, np.int32(n)) + ((np.int32(opt_stc), np.int32(opt_crs)) if opt_sfc==2 else
                                                  (np.int32(opt_stc),) if opt_crs==2 else ()),
    )
    return _unpack(cp, device_out, _vege.OUTPUT_NAMES)


def evaluate_vege_flux_masked_slab(fields: Mapping[str, object], n: int,
                                   active, *, use_device_libm: bool = False,
                            atan_profile="glibc_2_39") -> dict:
    """Evaluate only active columns in a fixed-width device slab.

    The leaf arithmetic is unchanged. Inactive outputs are positive zero,
    matching the former scatter allocation, and their physics is not evaluated.
    The live mask never determines a host allocation or launch dimension.
    """
    cp = _cupy()
    n = _check_size(n)
    active = cp.ascontiguousarray(active, dtype=cp.bool_)
    if active.shape != (n,):
        raise ValueError(f"VEGE_FLUX mask shape {active.shape} != {(n,)}")
    if n == 0:
        return {name: cp.empty(0, dtype=cp.float32)
                for name in _vege.OUTPUT_NAMES}
    packed = pack_vege_flux_slab(fields, n)
    device_out = cp.empty((n, _vege.N_OUTPUT), dtype=cp.float32)
    opt_stc = _stc_option(fields, "VEGE_FLUX")
    kernel_name = (f"k_vegeflux_masked_stc{opt_stc}" if opt_stc != 1
                   else "k_vegeflux_masked")
    opt_crs = _crs_option(fields)
    opt_sfc = _sfc_option(fields)
    if opt_crs == 2: kernel_name = "k_vegeflux_masked_crs2"
    if opt_sfc == 2: kernel_name = "k_vegeflux_masked_sfc2"
    profile_options = ({} if atan_profile == "glibc_2_39" else
                       {"atan_profile": atan_profile})
    module = (_vege._module(use_device_libm, selected_options=True, **profile_options)
              if opt_stc != 1 or opt_crs != 1 or opt_sfc != 1 else _vege._module(use_device_libm, **profile_options))
    kernel = module.get_function(kernel_name)
    args = (cp.ascontiguousarray(packed.inputs),
            cp.ascontiguousarray(packed.params),
            cp.ascontiguousarray(packed.isnow),
            cp.ascontiguousarray(packed.dzsnso),
            cp.ascontiguousarray(packed.stc),
            cp.ascontiguousarray(packed.df))
    if opt_stc == 3 or opt_crs == 2 or opt_sfc == 2:
        fsno = _column(cp, fields, "fsno", n, "VEGE_FLUX selected")
        args += (cp.ascontiguousarray(cp.broadcast_to(fsno, (n,)),
                                     dtype=cp.float32),)
    kernel(
        ((n + _vege.THREADS - 1) // _vege.THREADS,), (_vege.THREADS,),
        args + (device_out, np.int32(n), active) + ((np.int32(opt_stc), np.int32(opt_crs)) if opt_sfc==2 else
                                                  (np.int32(opt_stc),) if opt_crs==2 else ()),
    )
    return _unpack(cp, device_out, _vege.OUTPUT_NAMES)


# --------------------------------------------------------------------------
# BARE_FLUX
# --------------------------------------------------------------------------
def pack_bare_flux_slab(fields: Mapping[str, object], n: int):
    """Pack ``n`` land columns into ``noahmp_bare_flux``'s two arguments.

    Returns ``(inputs, ints)`` -- ``(n, 53) float32`` and ``(n, 5) int32``,
    the same two arrays :func:`woof.core.noahmp_bareflux_gpu.
    pack_bare_flux_calls` builds, in the same slot order.  ``urban_flag`` is
    reduced to the ``iurban`` slot by the same truthiness test the per-column
    packer applies; ``iurban`` is accepted as a spelling of it.
    """
    _slots_inputs = []
    _slots_ints = []
    cp = _cupy()
    n = _check_size(n)
    _check_options(fields, "BARE_FLUX", _BARE_OPTIONS)
    _stc_option(fields, "BARE_FLUX")

    opt_sfc = _sfc_option(fields)
    inputs = cp.empty((n, _bare.N_INPUT + (opt_sfc == 2)), dtype=cp.float32)
    for slot, name in enumerate(_bare.SCALAR_NAMES):
        _slots_inputs.append((slot, _column(cp, fields, name, n, "BARE_FLUX")))
    base = len(_bare.SCALAR_NAMES)
    for name in _bare.ARRAY_NAMES:
        _slots_inputs.append((base, _layers(cp, fields, name, n,
                                                "BARE_FLUX")))
        base += NLAYER

    if opt_sfc == 2:
        _slots_inputs.append((_bare.N_INPUT, _column(cp, fields, "CZIL", n, "BARE_FLUX SFC2")))
    ints = cp.empty((n, _bare.N_INT), dtype=cp.int32)
    for slot, name in enumerate(_bare.INT_NAMES):
        if name != "iurban":
            _slots_ints.append((slot, _int_column(cp, fields, name, n, "BARE_FLUX")))
            continue
        flag = fields["urban_flag"] if "urban_flag" in fields else _field(
            fields, "iurban", "BARE_FLUX")
        column = cp.asarray(flag)
        if column.ndim and column.shape != (n,):
            raise ValueError(
                f"BARE_FLUX slab field 'urban_flag' has shape "
                f"{tuple(column.shape)}; expected ({n},) or a scalar")
        _slots_ints.append((slot, (column != 0).astype(cp.int32)))
    copy_slab_slots(ints, _slots_ints)
    copy_slab_slots(inputs, _slots_inputs)
    return inputs, ints


def evaluate_bare_flux_slab(fields: Mapping[str, object], n: int) -> dict:
    """Run BARE_FLUX over a whole slab; return ``OUTPUT_NAMES -> (n,)``."""
    cp = _cupy()
    n = _check_size(n)
    if n == 0:
        return {name: cp.empty(0, dtype=cp.float32)
                for name in _bare.OUTPUT_NAMES}

    from woof.core.kernels import get_kernel

    inputs, ints = pack_bare_flux_slab(fields, n)
    device_out = cp.empty(n * _bare.N_OUTPUT, dtype=cp.float32)
    opt_stc = _stc_option(fields, "BARE_FLUX")
    kernel_name = (f"noahmp_bare_flux_stc{opt_stc}" if opt_stc != 1
                   else "noahmp_bare_flux")
    active = fields.get("active")
    if active is not None and kernel_name == "noahmp_bare_flux":
        kernel_name = "noahmp_bare_flux_masked"
    if active is not None and opt_stc == 2:
        kernel_name = "noahmp_bare_flux_masked_stc2"
    opt_sfc = _sfc_option(fields)
    if opt_sfc == 2:
        kernel_name = "noahmp_bare_flux_masked_sfc2" if active is not None else "noahmp_bare_flux_sfc2"
    if opt_sfc == 2 or kernel_name.endswith(("_stc2", "_stc3", "_masked")):
        from woof.core.noahmp_kernel_sources import selected_kernel
        kernel = selected_kernel("noahmp_bareflux", kernel_name)
    else:
        kernel = get_kernel("noahmp_bareflux", kernel_name)
    blocks = (n + _bare.THREADS - 1) // _bare.THREADS
    kernel(
        (blocks,), (_bare.THREADS,),
        (cp.ascontiguousarray(inputs).reshape(-1),
         cp.ascontiguousarray(ints).reshape(-1),
         device_out, np.int32(n), *((active,) if kernel_name in ("noahmp_bare_flux_masked", "noahmp_bare_flux_masked_stc2", "noahmp_bare_flux_masked_sfc2") else ()),
         *((np.int32(opt_stc),) if opt_sfc==2 else ())),
    )
    return _unpack(cp, device_out.reshape(n, _bare.N_OUTPUT),
                   _bare.OUTPUT_NAMES)


__all__ = [
    "LAYER_OFFSET",
    "NLAYER",
    "NSNOW",
    "NSOIL",
    "VegeFluxSlabInputs",
    "evaluate_bare_flux_slab",
    "evaluate_vege_flux_slab",
    "pack_bare_flux_slab",
    "pack_vege_flux_slab",
]
