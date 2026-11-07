"""WRF Registry surface feedback for the active SST/TMN options.

Masked REAL and INTEGER fields use copy_fcnm/copy_fcni, never averaging.
The source is WRF Registry.EM_COMMON and share/interp_fcn.F:1747-1906.
Counters and fields without a Registry ``u`` flag stay domain-local.
"""
from __future__ import annotations

import numpy as np


def feedback_fields(cfg):
    names = []
    if getattr(cfg, "sst_update", 0):
        names += ["xice", "xicem", "xland", "ivgtyp", "isltyp", "vegfra",
                  "tsk", "tmn", "snow", "snowc", "snowh", "tslb", "smois", "sh2o"]
    if getattr(cfg, "sst_skin", 0):
        names += ["sstsk", "tsk", "tslb"]
    if getattr(cfg, "tmn_update", 0):
        names += ["tmn", "tyr", "tyra", "tdly", "tlag"]
    return tuple(dict.fromkeys(names))


def _field(source, name):
    if source.store is not None:
        return source.array("fields/" + name)
    driver = getattr(source.state, "physics", None)
    return None if driver is None else driver.fields.get(name)


def commit_surface_feedback(parent_state, child_state, cfg, reg, *, spec_zone, skip_names=()):
    """Commit only active, allocated fields through bounded CUDA operands.

    Resident fields never leave the GPU. Streamed operands use the same
    admitted windows and canonical-store transaction as dynamical feedback.
    Returns transfer/scratch accounting for the enclosing transaction.
    """
    names = tuple(name for name in feedback_fields(cfg) if name not in skip_names)
    if not names:
        return {"fields": (), "sync_bytes": 0, "host_scratch_bytes": 0}
    from woof.core.nest_operands import (
        NestWindowSource, rectangle_windows, streamed_chunk_shape)
    from woof.core.nest_interp import feedback_parent_bounds, copy_fcnm, copy_fcni
    from woof.core.kernels import get_kernel
    import cupy as cp

    parent, child = NestWindowSource(parent_state), NestWindowSource(child_state)
    bounded = parent.store is not None or child.store is not None
    i0, i1, j0, j1 = feedback_parent_bounds(reg, spec_zone=spec_zone)
    if i0 > i1 or j0 > j1:
        return {"fields": (), "sync_bytes": 0, "host_scratch_bytes": 0}
    rectangle = (slice(j0, j1 + 1), slice(i0, i1 + 1))
    chunk = (streamed_chunk_shape(parent_state if parent.store is not None else child_state)
             if bounded else (reg.nyp, reg.nxp))
    committed = []
    # Mass-grid copy_fcnm's one-based child pick, converted to zero-based.
    oi = 1 if reg.nrj % 2 else reg.nri // 2 - 1
    oj = 1 if reg.nrj % 2 else reg.nrj // 2 - 1
    for name in names:
        target, donor = _field(parent, name), _field(child, name)
        if target is None or donor is None:
            # For heterogeneous domains only the common allocated state
            # has an operator. No history bank is invented on the parent.
            continue
        if (target.dtype != donor.dtype or target.dtype not in (np.float32, np.int32)
                or target.shape[:-2] != donor.shape[:-2]):
            raise ValueError(f"surface feedback {name} has incompatible state layouts")
        if not bounded:
            operator = copy_fcni if target.dtype == np.int32 else copy_fcnm
            operator(target, donor, reg, spec_zone=spec_zone)
        else:
            for window in rectangle_windows(rectangle, chunk):
                y, x = window
                ci = (x.start + 1 - reg.i_parent_start) * reg.nri + oi
                cj = (y.start + 1 - reg.j_parent_start) * reg.nrj + oj
                ny, nx = y.stop-y.start, x.stop-x.start
                donor_window = (slice(cj, cj+(ny-1)*reg.nrj+1),
                                slice(ci, ci+(nx-1)*reg.nri+1))
                source = child.device_array(donor, donor_window)
                out = cp.empty((*target.shape[:-2], ny, nx), dtype=target.dtype)
                get_kernel("surface_analysis", "surface_feedback_pick")(
                    ((out.size+255)//256,), (256,),
                    (out, source, *(np.int32(v) for v in
                     (nx, ny, source.shape[-1], source.shape[-2], reg.nri, reg.nrj, out.size))))
                if parent.store is not None:
                    parent.write("fields/"+name, window, out)
                else:
                    target[(...,)+window] = out
        committed.append(name)
    return {"fields": tuple(committed),
            "sync_bytes": parent.host_to_device_bytes + child.host_to_device_bytes + parent.device_to_host_bytes,
            "host_scratch_bytes": max(parent.max_host_scratch_bytes, child.max_host_scratch_bytes)}
