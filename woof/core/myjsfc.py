"""MYJ (Eta similarity) surface layer, ``sf_sfclay_physics = 2``.

CUDA launcher around :mod:`woof.core.kernels.myjsfc`, the device port of
the byte-frozen WRF v4.6.1 ``phys/module_sf_myjsfc.F`` (the float32 CPU
authority ``woof.verify.myj_ref.np_myjsfc_column`` is its tolerance
twin).  Both interpolate the SAME similarity tables: they are built once on
the host by :func:`woof.core.myjsfc_tables.build_psi_tables` (a
transcription of ``MYJSFCINIT``, phys/module_sf_myjsfc.F:1174-1299) and
uploaded here.

The scheme is the Eta surface layer WRF pairs with the MYJ PBL and with
nothing else (phys/module_physics_init.F:3770-3772 fatals a MYJ PBL whose
surface layer did not set ``isfc = 2``).  woof enforces the same pairing in
:func:`woof.config.validate_myj_pairing`.

Conformance status: BIT-IDENTICAL to WRF v4.6.1's MYJSFC and MYJSFCINIT
on the 224-column oracle (tools/myjsfc_wrf461_oracle,
tests/test_myjsfc_wrf461_parity.py) under the strict build and under
default arithmetic; registry maturity stays implemented-unverified because
no matched forecast trajectory exists.  ``surface["ht"]`` is WRF's HT:
MYJSFC's interface heights start at the terrain height (ZINT(KTE+1)=HT,
:165), PBLH and ZSL are differences of them, and the float32 words depend
on it.
"""

from __future__ import annotations

from woof.core.device_cache import cuda_cache

import cupy as cp
import numpy as np

from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.myjsfc_tables import build_psi_tables
# WRF INOUT surface state and pure outputs, in kernel argument order.  The
# tables live in the runtime-free inventory module so the VRAM estimator
# prices the same names this launcher binds without importing cupy.
from woof.core.physics_inventory import (MYJ_SFCLAY_INOUT,
                                          MYJ_SFCLAY_OUTPUTS)
from woof.core.state import DTYPE

_TPB = 128

#: Read-only column inputs, in kernel argument order.
_COLUMN_INPUTS = ("dz", "tke")
#: Read-only surface inputs, in kernel argument order.
_SURFACE_INPUTS = ("u1", "v1", "t1", "th1", "qv1", "qc1", "p1", "psfc",
                   "tsk", "xland", "mavail", "z0base", "ht")


@cuda_cache(maxsize=None, ready=True)
def _device_tables(correctly_rounded_atan: bool = False):
    """Upload the MYJSFCINIT tables once per process and ATAN generation.

    The scalars ride along as Python floats; the kernel takes them by
    value.  ``lru_cache`` keeps the four 10001-word arrays resident for the
    life of the process rather than re-uploading 160 kB every call.
    ``correctly_rounded_atan`` is the referee WRF's libm generation
    (:func:`woof.wrf_exact.referee_correctly_rounded_trig`).
    """
    tables = build_psi_tables(correctly_rounded_atan)
    arrays = tuple(cp.asarray(tables[name], dtype=DTYPE)
                   for name in ("psim1", "psih1", "psim2", "psih2"))
    scalars = tuple(DTYPE(tables[name]) for name in (
        "dzeta1", "dzeta2", "ztmin1", "ztmax1", "ztmin2", "ztmax2",
        "fh01", "fh02"))
    return arrays, scalars


def launch_myj_sfclay(columns, surface, state, outputs, *,
                      itimestep: int, ocean_z0=None) -> None:
    """Run one MYJ surface-layer step in place.

    ``columns`` supplies the full bottom-up ``(nz, ny, nx)`` ``dz`` and
    ``tke`` arrays the PBLH scan needs (module_sf_myjsfc.F:263-277);
    ``surface`` the lowest-model-level and static fields; ``state`` the
    nine WRF INOUT values; ``outputs`` the pure outputs.  Every array is
    contiguous float32 with the same ``(ny, nx)`` surface shape.
    """
    shape = surface["psfc"].shape
    nz = int(columns["dz"].shape[0])
    if nz < 2:
        raise ValueError("the MYJ surface layer needs at least two levels: "
                         "its PBLH scan walks the TKE column")
    for name in _COLUMN_INPUTS:
        array = columns[name]
        if (array.shape != (nz, *shape) or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError(
                f"launch_myj_sfclay needs a contiguous float32 "
                f"({nz}, {shape[0]}, {shape[1]}) {name!r} column")
    named = (
        [columns[name] for name in _COLUMN_INPUTS]
        + [surface[name] for name in _SURFACE_INPUTS]
        + [state[name] for name in MYJ_SFCLAY_INOUT]
        + [outputs[name] for name in MYJ_SFCLAY_OUTPUTS])
    for array in named[len(_COLUMN_INPUTS):]:
        if (array.shape != shape or array.dtype != DTYPE
                or not array.flags.c_contiguous):
            raise ValueError("launch_myj_sfclay requires same-shape "
                             "contiguous float32 surface arrays")
    ocean_args = ()
    if ocean_z0 is not None:
        if (not isinstance(ocean_z0, cp.ndarray)
                or ocean_z0.shape != shape or ocean_z0.dtype != DTYPE
                or not ocean_z0.flags.c_contiguous
                or ocean_z0.device.id != surface["psfc"].device.id):
            raise ValueError("MYJ wave roughness requires contiguous float32 on the surface device/grid")
        ocean_args = (ocean_z0,)
    from woof.wrf_exact import referee_correctly_rounded_trig
    arrays, scalars = _device_tables(referee_correctly_rounded_trig())
    n = int(np.prod(shape))
    blocks = (n + _TPB - 1) // _TPB
    kernel = (get_kernel_int_defines("myjsfc", "myjsfc_column",
              (("OCEAN_WAVE_ROUGHNESS", 1),)) if ocean_z0 is not None
              else get_kernel("myjsfc", "myjsfc_column"))
    kernel((blocks,), (_TPB,), tuple(named) + arrays + scalars + ocean_args
           + (np.int32(itimestep), np.int32(nz), np.int32(n)))
