"""Explicit replay ABI for the native NWP window arithmetic.

The operational driver in nwp_diagnostics binds individual persistent planes;
this stacked ABI lets the oracle compare the unchanged native module output.
"""
from __future__ import annotations
import math
import numpy as np
from woof.core.level_diagnostics import _array

FIELDS = ("WSPD10MAX", "UP_HELI_MAX", "W_UP_MAX", "W_DN_MAX", "W_MEAN", "GRPL_MAX",
          "REFD_MAX", "HAIL_MAXK1", "HAIL_MAX2D", "W_COLMEAN", "NUMCOLPTS", "GRPL_COLINT")


def update(w, p8w, znw, qg, rho, ph, phb, u10, v10, refl, output, *, dt,
           xtime, history_minutes, step, adaptive=False, max_time_step=0):
    """Launch in the caller's stream; all arrays must already be on device.

    ``xtime`` is native forecast minutes, ``step`` is native itimestep.
    ``output`` persists across calls. Hail/UH planes are reset when native
    resets them, but their production is the other native routines' duty.
    """
    nz, ny, nx = qg.shape
    if (not math.isfinite(dt) or dt <= 0 or not math.isfinite(xtime)
            or history_minutes < 1 or int(history_minutes) != history_minutes
            or np.float32(history_minutes * 60.) / np.float32(dt) < 1
            or step < 1 or int(step) != step):
        raise ValueError("NWP windows require finite dt>0, integer history_minutes>0, dt<=interval and integer step>=1")
    for name, value in (("w", w), ("p8w", p8w), ("ph", ph), ("phb", phb)):
        _array(value, (nz + 1, ny, nx), name)
    _array(znw, (nz + 1,), "znw")
    for name, value in (("qg", qg), ("rho", rho), ("refl", refl)):
        _array(value, (nz, ny, nx), name)
    for name, value in (("u10", u10), ("v10", v10)):
        _array(value, (ny, nx), name)
    _array(output, (len(FIELDS), ny, nx), "output")
    from woof.core.kernels import get_kernel
    get_kernel("nwp_windows", "nwp_windows")(((ny * nx + 127) // 128,), (128,),
        (w, p8w, znw, qg, rho, ph, phb, u10, v10, refl, output,
         np.int32(nz), np.int32(ny), np.int32(nx), np.float32(dt), np.float32(xtime),
         np.int32(history_minutes), np.int32(step), np.int32(adaptive), np.int32(max_time_step)))
