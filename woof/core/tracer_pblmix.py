"""Active native tracer diff4d right hand sides sharing one factorization."""
from __future__ import annotations

import numpy as np

from woof.core.kernels import get_kernel
from woof.core.passive_tracer_contract import TRACER_NAMES


def tracer_pblmix_rates(state, dz, rho, exch_h, dt):
    """Return raw native rates; dry-mass coupling stays with the PBL driver.

    Scratch and pointer carriers use the state's arena, so allocation,
    streaming and device inventory see them. No device value is read here.
    """
    import cupy as cp
    shape = tuple(exch_h.shape)
    nz, ny, nx = shape
    names = tuple(name for name in TRACER_NAMES if getattr(state, name, None) is not None)
    fields = [getattr(state, name) for name in names]
    count_fields = len(names)
    if any(a.shape != shape or a.dtype != cp.float32 or not a.flags.c_contiguous
           for a in (*fields, dz, rho, exch_h)):
        raise ValueError("tracer diff4d requires native contiguous float32 mass carriers")
    pointers = state.scratch((2*count_fields,), "tracer_pblmix_pointers").view(cp.uint64)
    identity = (pointers.data.ptr, *(a.data.ptr for a in fields))
    if getattr(state, "_tracer_pblmix_pointer_identity", None) != identity:
        pointers.set(np.asarray(identity[1:], dtype=np.uint64))
        state._tracer_pblmix_pointer_identity = identity
    rates = state.scratch((count_fields, *shape), "tracer_pblmix_rates")
    ncol = ny * nx
    work = state.scratch(((5*nz+1)*min(ncol, 32768),), "tracer_pblmix_work")
    for first in range(0, ncol, 32768):
        count = min(ncol-first, 32768)
        get_kernel("tracer_pblmix", "tracer_pblmix_columns")(
            ((count+127)//128,), (128,),
            (pointers, dz, rho, exch_h, np.float32(dt), rates, work,
             np.int32(nz), np.int32(ncol), np.int32(first), np.int32(count), np.int32(count_fields)))
    return dict(zip(names, rates, strict=True))
