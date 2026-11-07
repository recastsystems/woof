"""WRF MM5 surface-layer options 91 (classic) and 1 (revised).

The CUDA kernel consumes lowest-model-level fields on mass points, one
thread per ``(j, i)`` surface column.  It mirrors WRF v4.6.1
``module_sf_sfclay.F:SFCLAY1D`` and
``physics_mmm/sf_sfclayrev.F90:sf_sfclayrev_run``.  ``xland`` uses WRF's
convention (1 land, 2 water); ``znt``, ``ust``, ``ustm``, ``mol``, ``hfx``,
``qfx``, ``qsfc`` and ``zol`` are copied as previous-step/inout values
before launch.

The high-level :func:`sfclay` allocates and returns every diagnostic and
exchange field needed by the later LSM/PBL driver.  :func:`launch_sfclay`
is the allocation-free form.  The float64 verification authority is
``woof.verify.npref.np_sfclay``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel
from woof.core.state import DTYPE


from woof.core.physics_inventory import SFCLAY_OUTPUTS  # noqa: F401  (one home; re-exported here)


@dataclass
class SFClayResult:
    """FP32 device outputs of one surface-layer call."""

    znt: cp.ndarray
    ust: cp.ndarray
    ustm: cp.ndarray
    mol: cp.ndarray
    hfx: cp.ndarray
    qfx: cp.ndarray
    qsfc: cp.ndarray
    zol: cp.ndarray
    regime: cp.ndarray
    psim: cp.ndarray
    psih: cp.ndarray
    fm: cp.ndarray
    fh: cp.ndarray
    lh: cp.ndarray
    u10: cp.ndarray
    v10: cp.ndarray
    th2: cp.ndarray
    t2: cp.ndarray
    q2: cp.ndarray
    chs: cp.ndarray
    chs2: cp.ndarray
    cqs2: cp.ndarray
    flhc: cp.ndarray
    flqc: cp.ndarray
    qgh: cp.ndarray
    rmol: cp.ndarray
    wspd: cp.ndarray
    br: cp.ndarray
    gz1oz0: cp.ndarray
    cpm: cp.ndarray
    ck: cp.ndarray
    cka: cp.ndarray
    cd: cp.ndarray
    cda: cp.ndarray


_TPB = 128


def _validate_options(option: int, isftcflx: int, iz0tlnd: int) -> None:
    if option not in (1, 91):
        raise ValueError(f"sfclay option must be 1 or 91, got {option}")
    if isftcflx not in (0, 1, 2):
        raise ValueError(f"isftcflx must be 0, 1, or 2, got {isftcflx}")
    if iz0tlnd not in (0, 1, 2):
        raise ValueError(f"iz0tlnd must be 0, 1, or 2, got {iz0tlnd}")


def _surface_array(value, shape, name: str, default: float | None = None):
    if value is None:
        if default is None:
            raise TypeError(f"{name} is required")
        return cp.full(shape, default, dtype=DTYPE)
    array = cp.asarray(value, dtype=DTYPE)
    if array.shape != shape:
        try:
            array = cp.broadcast_to(array, shape)
        except ValueError as exc:
            raise ValueError(f"{name} shape {array.shape} is not broadcastable "
                             f"to surface shape {shape}") from exc
    return cp.ascontiguousarray(array)


def _allocate_result(shape, *, znt, ust, ustm, mol, hfx, qfx, qsfc, zol):
    initial = {"znt": znt.copy(), "ust": ust.copy(), "ustm": ustm.copy(),
               "mol": mol.copy(), "hfx": hfx.copy(), "qfx": qfx.copy(),
               "qsfc": qsfc.copy(), "zol": zol.copy()}
    # A first flux-off call preserves these buffers.  Seed the allocating
    # API's cold start, rather than letting it preserve uninitialised memory.
    initial.update({name: cp.zeros(shape, dtype=DTYPE)
                    for name in ("chs", "chs2", "cqs2")})
    arrays = {name: initial.get(name, cp.empty(shape, dtype=DTYPE))
              for name in SFCLAY_OUTPUTS}
    return SFClayResult(**arrays)


def launch_sfclay(u, v, t, qv, p, dz8w, psfc, tsk, pblh, mavail, xland,
                   lakemask, result: SFClayResult, *, option: int, dx: float,
                   isfflx: bool = True, isftcflx: int = 0,
                   iz0tlnd: int = 0, dx_field=None) -> None:
    """Launch into a preallocated :class:`SFClayResult`.

    All inputs and result fields must be contiguous FP32 arrays with the same
    ``(ny,nx)`` shape.  WRF inout values are already held in
    ``result.znt/ust/ustm/mol/hfx/qfx/qsfc/zol``.  With ``isfflx=False``,
    ``result.chs/chs2/cqs2`` also retain their incoming values; HFX, QFX,
    FLHC, FLQC and WOOF's defined LH are zero.
    """
    _validate_options(option, isftcflx, iz0tlnd)
    shape = u.shape
    arrays = (u, v, t, qv, p, dz8w, psfc, tsk, pblh, mavail, xland,
              lakemask) + tuple(getattr(result, name) for name in SFCLAY_OUTPUTS)
    for array in arrays:
        if array.shape != shape or array.dtype != DTYPE or not array.flags.c_contiguous:
            raise ValueError("launch_sfclay requires same-shape contiguous "
                             "float32 surface arrays")
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    if dx_field is not None:
        if dx_field.shape != shape or dx_field.dtype != DTYPE or not dx_field.flags.c_contiguous:
            raise ValueError("dx_field must be a contiguous float32 surface plane")
    kernel = get_kernel("sfclay_dx2d" if dx_field is not None else "sfclay", "sfclay_column")
    # Kernel order: 12 read-only inputs; 8 inout fields; remaining outputs.
    kernel((blocks,), (_TPB,), arrays + (
        dx_field if dx_field is not None else DTYPE(dx), np.int32(option), np.int32(bool(isfflx)),
        np.int32(isftcflx), np.int32(iz0tlnd), np.int32(n)))


def sfclay(u, v, t, qv, p, dz8w, psfc, tsk, znt, pblh, mavail, xland,
           *, option: int = 91, qsfc=None, zol=None, ust=None, ustm=None,
           mol=None, hfx=None, qfx=None, lakemask=None, dx: float = 1000.0,
           isfflx: bool = True, isftcflx: int = 0,
           iz0tlnd: int = 0) -> SFClayResult:
    """Run one WRF MM5 surface-layer call and return FP32 device fields.

    Required fields are broadcast-compatible 2-D surface arrays.  Defaults
    match a first WRF call: ``ust=0.1``, ``zol/mol/hfx/qfx/qsfc=0`` and no
    lake mask.  ``ustm`` defaults to 0 because Registry.EM_COMMON:1954
    declares it plain state and no WRF init routine seeds it.  Classic
    option 91 preserves incoming ``zol`` in its strong-stable branch,
    matching WRF's inout semantics.  ``option`` is 91
    (classic) or 1 (revised); configuration value 0 means the future physics
    driver must skip this function entirely.
    """
    _validate_options(option, isftcflx, iz0tlnd)
    u = cp.ascontiguousarray(cp.asarray(u, dtype=DTYPE))
    if u.ndim != 2:
        raise ValueError(f"sfclay surface inputs must be 2-D (ny,nx), got {u.shape}")
    shape = u.shape
    base = [u] + [_surface_array(value, shape, name) for name, value in (
        ("v", v), ("t", t), ("qv", qv), ("p", p), ("dz8w", dz8w),
        ("psfc", psfc), ("tsk", tsk), ("pblh", pblh), ("mavail", mavail),
        ("xland", xland))]
    lake = _surface_array(lakemask, shape, "lakemask", 0.0)
    znt_a = _surface_array(znt, shape, "znt")
    ust_a = _surface_array(ust, shape, "ust", 0.1)
    ustm_a = _surface_array(ustm, shape, "ustm", 0.0)
    mol_a = _surface_array(mol, shape, "mol", 0.0)
    hfx_a = _surface_array(hfx, shape, "hfx", 0.0)
    qfx_a = _surface_array(qfx, shape, "qfx", 0.0)
    qsfc_a = _surface_array(qsfc, shape, "qsfc", 0.0)
    zol_a = _surface_array(zol, shape, "zol", 0.0)
    result = _allocate_result(shape, znt=znt_a, ust=ust_a, ustm=ustm_a,
                              mol=mol_a, hfx=hfx_a, qfx=qfx_a, qsfc=qsfc_a,
                              zol=zol_a)
    launch_sfclay(*base, lake, result, option=option, dx=dx,
                   isfflx=isfflx, isftcflx=isftcflx, iz0tlnd=iz0tlnd)
    return result
