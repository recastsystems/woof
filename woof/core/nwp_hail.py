"""CUDA implementation of the native WRF 4.7.1 NWP PSD hail calculation.

The nwp_diagnostics driver supplies its native density and interval carriers.
HAILCAST remains a separately selected stone model.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from woof.core.level_diagnostics import _array


@dataclass(frozen=True)
class NsslSingleMoment:
    cnoh: float
    cnohl: float
    rho_qh: float
    rho_qhl: float
    alphah: float
    alphahl: float
    hail: int = 0


def update(qg, rho, hail1, hail2, *, mp_physics, qh=None, ng=None, nh=None,
           volume=None, qr=None, nr=None, mpuse_hail=0, gsfcgce_hail=0,
           gsfcgce_2ice=0, nssl_single_moment=None):
    """Update existing maxima in metres, in the caller's CUDA stream.

    Call the native window reset before this leaf. Species presence is
    meaningful even when its entire array is zero, exactly as Fortran PRESENT.
    ``mpuse_hail`` is the native driver's scheme-specific translated value.
    """
    nz, ny, nx = qg.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("NWP hail requires nonempty columns")
    if volume is not None and mp_physics != 38:
        raise ValueError("native NWP only initializes the volume density index under THOMPSONGH (38)")
    if mp_physics in (8, 28) and ng is None and (qr is None or nr is None):
        raise ValueError("native Thompson NWP branch reads QR and NR when NG is absent")
    args = []
    for name, value in (("qg", qg), ("rho", rho), ("qh", qh), ("ng", ng), ("nh", nh),
                        ("volume", volume), ("qr", qr), ("nr", nr)):
        if value is not None:
            _array(value, (nz, ny, nx), name)
        args.append(qg if value is None else value)
    for name, value in (("hail1", hail1), ("hail2", hail2)):
        _array(value, (ny, nx), name)
    scalars = [0.] * 6
    nssl_hail = 0
    if nssl_single_moment is not None:
        if mp_physics != 18:
            raise ValueError("NSSL single-moment constants require mp_physics=18")
        scalars = [getattr(nssl_single_moment, k) for k in
                   ("cnoh", "cnohl", "rho_qh", "rho_qhl", "alphah", "alphahl")]
        if not all(math.isfinite(v) for v in scalars):
            raise ValueError("NSSL constants must be finite")
        nssl_hail = nssl_single_moment.hail
    from woof.core.kernels import get_kernel
    get_kernel("nwp_hail", "nwp_hail")(((ny*nx+63)//64,), (64,),
        (*args, hail1, hail2, np.int32(nz), np.int32(ny*nx), np.int32(mp_physics),
         *(np.int32(v is not None) for v in (qh, ng, nh, volume)),
         np.int32(mpuse_hail), np.int32(gsfcgce_hail), np.int32(gsfcgce_2ice),
         np.int32(nssl_single_moment is None), np.int32(nssl_hail),
         *(np.float32(v) for v in scalars)))
