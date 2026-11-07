"""SSiB's WRF Registry feedback, including bounded canonical stores.

The authority is WRF v4.6.1 Registry.EM_COMMON's generated
nest_feedbackup_interp.inc. Its SSiB private state has no feedback flag.
Surface fields here have no smoothing flag either.
"""
from __future__ import annotations

import numpy as np

PICK_FIELDS = ('smois', 'tslb', 'snow', 'canwat', 'tsk', 'snowh')
MEAN_FIELDS = ('t2', 'q2', 'u10', 'v10')


def feedback(parent, child, reg, *, spec_zone=1, chunk_shape=None):
    import cupy as cp
    from woof.core.kernels import get_kernel
    from woof.core.nest_interp import (
        copy_fcn, feedback_child_window, feedback_parent_bounds)
    from woof.core.nest_operands import NestWindowSource, rectangle_windows

    def fields(state):
        owner = getattr(state, '_streamed_domain', None)
        if owner is not None:
            return {n: owner.store[f'fields/{n}']
                    for n in PICK_FIELDS + MEAN_FIELDS}
        return state.physics.fields

    pf, cf = fields(parent), fields(child)
    ps, cs = NestWindowSource(parent), NestWindowSource(child)
    ilo, ihi, jlo, jhi = feedback_parent_bounds(reg, spec_zone=spec_zone)
    if ihi < ilo or jhi < jlo:
        return
    sy, sx = chunk_shape or (reg.nyc, reg.nxc)
    chunks = (max(1, sy // reg.nrj), max(1, sx // reg.nri))
    pick = get_kernel('ssib_records', 'ssib_feedback_pick')
    for name in PICK_FIELDS + MEAN_FIELDS:
        target, source = pf[name], cf[name]
        for win in rectangle_windows((slice(jlo,jhi+1),slice(ilo,ihi+1)), chunks):
            donor = feedback_child_window(reg, win, spec_zone=spec_zone)
            incoming = cs.device_array(source, donor)
            result = cp.empty(target[(...,)+win].shape, dtype=cp.float32)
            if name in MEAN_FIELDS:
                copy_fcn(result, incoming, reg, spec_zone=spec_zone,
                         parent_window=win, child_window=donor)
            else:
                py, px = win; cy, cx = donor
                args = (result.size, *result.shape[-2:], *incoming.shape[-2:],
                        px.start, py.start, cx.start, cy.start,
                        reg.i_parent_start, reg.j_parent_start, reg.nri, reg.nrj)
                pick(((result.size+255)//256,), (256,),
                     (result, incoming, *(np.int32(v) for v in args)))
            if isinstance(target, np.ndarray):
                target[(...,)+win] = cp.asnumpy(result)
                ps.device_to_host_bytes += result.nbytes
            else:
                target[(...,)+win] = result
    return cs.host_to_device_bytes + ps.device_to_host_bytes
