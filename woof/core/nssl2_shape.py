"""CUDA specialization of the public NSSL gamma-distribution shapes.

Default shapes retain the existing compiled images.  Coefficient words are
compile-time constants, so no coefficient arrays or host reads enter a step.
"""
from __future__ import annotations

import struct
import math
from woof.core.device_cache import cuda_cache

DEFAULT_SHAPES = (0.0, 1.0)
_MODULES = {
    "nssl2_nucond": "nssl2_shape_nucond",
    "nssl2_fused_gs": "nssl2_shape_gs",
    "nssl2_driver_support": "nssl2_shape_driver",
    "nssl2_diagnostics": "nssl2_shape_diagnostics",
}


def resolved_shapes(nssl_alphah=0.0, nssl_alphahl=1.0, nssl_icdx=6, nssl_icdxhl=6):
    """Native REAL coefficients with the module's gamma-table address bounds.

    WRF has no namelist interval check. Its gmoi(0:3001) interpolation reads
    both i and i+1, i=INT(100*argument). The largest setvtz argument for the
    default MM drag table is (7+alpha)+0.67819; the smallest is 1+alpha.
    Reject singular distributions and out-of-array addresses, without
    treating the incomplete-gamma table's alpha=15 clamp as a namelist cap.
    """
    import numpy as np
    result = []
    for name, value in (("nssl_alphah", nssl_alphah),
                        ("nssl_alphahl", nssl_alphahl)):
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                a = np.float32(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite native REAL") from exc
        if not math.isfinite(float(a)) or a <= -1.0:
            raise ValueError(f"{name} must define a finite gamma distribution (alpha > -1)")
        arg = np.float32(np.float32(np.float32(7.0) + a) + np.float32(0.67819))
        if int(np.float32(np.float32(100.0) * arg)) + 1 > 3001:
            raise ValueError(f"{name} exceeds native gmoi(0:3001) interpolation addresses "
                             "(module_mp_nssl_2mom.F:798, :7464-7472)")
        result.append(float(a))
    drags = resolved_drags(nssl_icdx, nssl_icdxhl)
    return tuple(result) if drags == (6, 6) else (*result, *drags)


def resolved_drags(nssl_icdx=6, nssl_icdxhl=6):
    from numbers import Integral
    values = (nssl_icdx, nssl_icdxhl)
    for name, value in zip(("nssl_icdx", "nssl_icdxhl"), values, strict=True):
        if isinstance(value, bool) or not isinstance(value, Integral) or not -(2**31) <= value < 2**31:
            raise ValueError(f"{name} must be a native 32-bit INTEGER")
    return tuple(int(v) for v in values)


def shape_defines(shapes):
    alpha = tuple((name, struct.unpack("<I", struct.pack("<f", value))[0] + 1)
                 for name, value in zip(
                     ("NSSL2_ALPHA_G_WORD", "NSSL2_ALPHA_H_WORD"), shapes[:2],
                     strict=True))
    drag = () if len(shapes) == 2 else tuple((name, (value & 0xffffffff) + 1) for name, value in zip(("NSSL2_DRAG_G_WORD", "NSSL2_DRAG_H_WORD"), shapes[2:], strict=True))
    return (*alpha, *drag)


@cuda_cache(maxsize=None)
def shape_kernel(module, function, shapes=DEFAULT_SHAPES, defines=()):
    from woof.core.kernels import get_kernel, get_kernel_int_defines
    if module == "nssl2_diagnostics":
        shapes = tuple(shapes[:2])  # Drag does not enter radar.
    if tuple(shapes) != DEFAULT_SHAPES:
        module = _MODULES[module]
        defines = (*defines, *shape_defines(shapes))
    if defines:
        return get_kernel_int_defines(module, function, tuple(sorted(defines)))
    return get_kernel(module, function)


def requires_cd_carry(shapes):
    g,h=(6,6) if len(shapes)==2 else shapes[2:]
    return g>6 or h==2 or h>6
