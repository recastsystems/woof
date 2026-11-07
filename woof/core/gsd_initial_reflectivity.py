"""The pinned HRRR fork's initial-history Thompson radar helper."""
from __future__ import annotations

import numpy as np

MODULE = "gsd_initial_reflectivity"
KERNEL = "gsd_refl_first_time_step"
BLOCK = 64
FLOAT_SLOTS, DOUBLE_SLOTS, BOOL_SLOTS = 13, 4, 3


def enabled(cfg):
    return bool(getattr(cfg, "gsd_diagnostics", 0)
                and getattr(cfg, "mp_physics", 0) in (8, 28)
                and getattr(cfg, "do_radar_ref", False))


def tile_threads(multiprocessors, columns):
    return max(1, min((int(columns)+BLOCK-1)//BLOCK, int(multiprocessors)*4))*BLOCK


def workspace_bytes(nz, threads):
    return (int(nz)+1)*int(threads)*(FLOAT_SLOTS*4+DOUBLE_SLOTS*8+BOOL_SLOTS)


def launch(th, pii, p, rho, qv, qc, nr, qr, qs, qg, refl, *, state_inputs=False, use_theta_m=0):
    """Read-only native inputs; with state_inputs, th/pii/rho mean thb/thp/alt.

    The fallback is applied to a private column, never to prognostic NR.
    The arena is bounded by SM count rather than full-domain column count.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    shape = tuple(qv.shape)
    nz, ny, nx = shape
    for name, array in zip(("pii", "p", "rho", "qv", "qc", "nr", "qr", "qs", "qg", "refl"),
                           (pii, p, rho, qv, qc, nr, qr, qs, qg, refl)):
        if array.shape != shape or array.dtype != np.float32 or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous float32 {shape}")
    if th.shape not in ((shape, (nz,)) if state_inputs else (shape,)) or th.dtype != np.float32 or not th.flags.c_contiguous:
        raise ValueError("th must be a contiguous float32 full field or state base profile")
    threads = tile_threads(cp.cuda.Device().attributes["MultiProcessorCount"], ny*nx)
    cells = (nz+1)*threads
    f = cp.empty(FLOAT_SLOTS*cells, cp.float32)
    d = cp.empty(DOUBLE_SLOTS*cells, cp.float64)
    b = cp.empty(BOOL_SLOTS*cells, cp.bool_)
    if use_theta_m:
        from woof.core.kernels import get_kernel_int_defines
        kernel = get_kernel_int_defines(MODULE, KERNEL, (("GPUWM_USE_THETA_M", 1),))
    else:
        kernel = get_kernel(MODULE, KERNEL)
    kernel((threads//BLOCK,), (BLOCK,),
        (th, pii, p, rho, qv, qc, nr, qr, qs, qg, refl, f, d, b,
         np.int32(nz), np.int32(ny*nx), np.int32(state_inputs), np.int32(th.ndim == 3)))


def diagnose(state, cfg):
    if cfg.mp_physics not in (8, 28):
        raise ValueError("gsd_refl_first_time_step supports Thompson (8/28) only")
    refl = state.scratch(state.qv.shape, "refl_10cm")
    launch(state.thb, state.thp, state.p, state.alt, state.qv, state.qc,
           state.nr, state.qr, state.qs, state.qg, refl, state_inputs=True,
           use_theta_m=getattr(state, "use_theta_m", 0))
    return refl
