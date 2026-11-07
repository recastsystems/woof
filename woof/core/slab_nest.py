"""WRF slab surface feedback, including bounded canonical tile stores.

Registry.EM_COMMON:790,1417 assigns TSLB/TSK to copy_fcnm; :1964,1967
assigns ACHFX/ACLHF to copy_fcn. None carries the smoothing flag. SFCEVP,
CAPG, THC and GSW carry no feedback flag. The native copy_fcnm argument
imask is unused (share/interp_fcn.F:1747-1824).
"""
from __future__ import annotations

import numpy as np


def feedback_slab_surface(parent, child, reg, *, spec_zone=1,
                          chunk_shape=None):
    """Restrict only the slab's Registry feedback fields, on the device."""
    from woof.core.nest_interp import (
        copy_fcn, feedback_child_window, feedback_parent_bounds)
    from woof.core.nest_operands import NestWindowSource
    import cupy as cp

    sources = [NestWindowSource(state) for state in (parent, child)]

    def field(source, name):
        if source.store is not None:
            return source.array(f"fields/{name}")
        return source.state.physics.fields[name]

    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=spec_zone)
    sy, sx = chunk_shape or (reg.nyc, reg.nxc)
    py, px = max(1, sy // reg.nrj), max(1, sx // reg.nri)
    for name in ("tsk", "tslb", "achfx", "aclhf"):
        target, source = (field(s, name) for s in sources)
        if target.shape[:-2] != source.shape[:-2]:
            raise ValueError("WRF slab feedback requires the same soil layer count on both domains")
        for j in range(jlo, jhi + 1, py):
            for i in range(ilo, ihi + 1, px):
                window = (slice(j, min(j + py, jhi + 1)),
                          slice(i, min(i + px, ihi + 1)))
                view = target[(...,) + window]
                if name in ("tsk", "tslb"):
                    # WRF's odd branch adds istag/jstag, even branch
                    # adds ratio/2-1. Preserve that literal rule for
                    # ratios other than 3 too, without a host gather.
                    oy = 1 if reg.nrj % 2 else reg.nrj // 2 - 1
                    ox = 1 if reg.nrj % 2 else reg.nri // 2 - 1
                    y0 = (j + 1 - reg.j_parent_start) * reg.nrj + oy
                    x0 = (i + 1 - reg.i_parent_start) * reg.nri + ox
                    y1 = y0 + (window[0].stop - j - 1) * reg.nrj + 1
                    x1 = x0 + (window[1].stop - i - 1) * reg.nri + 1
                    donor = (slice(y0, y1), slice(x0, x1))
                    values = sources[1].device_array(source, donor)
                    result = values[..., ::reg.nrj, ::reg.nri]
                else:
                    donor = feedback_child_window(reg, window, spec_zone=spec_zone)
                    values = sources[1].device_array(source, donor)
                    result = cp.empty(view.shape, dtype=cp.float32)
                    copy_fcn(result, values, reg, spec_zone=spec_zone,
                             parent_window=window, child_window=donor)
                if isinstance(view, np.ndarray):
                    view[...] = cp.asnumpy(result)
                else:
                    view[...] = result


__all__ = ["feedback_slab_surface"]
