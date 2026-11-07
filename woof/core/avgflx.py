"""WRF 4.6.1 Registry/registry.avgflx and module_avgflx_em.F.

The means remain readable after history output. The next completed step
resets them before adding its final RK acoustic mean. Count and reset flag
are replicated per column so tiled and ranked transport use the ordinary
mass-grid carrier machinery without a host scalar transfer at every step.
"""
from __future__ import annotations

import numpy as np

from woof.avgflx_contract import HISTORY, PERSISTENT, CUGD_REFUSAL


def scratch_shapes(cfg):
    if not cfg.do_avgflx_em:
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    return dict(avgflx_rum=(nz, ny, nx + 1),
                avgflx_rvm=(nz, ny + 1, nx),
                avgflx_wwm=(nz + 1, ny, nx),
                avgflx_clock=(2, ny, nx))


def initialize(state, cfg):
    for slot, shape in scratch_shapes(cfg).items():
        state.scratch(shape, slot,
                      dtype=np.int32 if slot == "avgflx_clock" else np.float32)


def update(state, cfg, fluxes):
    """Device-only running mean; called once after the final RK stage."""
    from woof.core.kernels import get_kernel

    clock = state.existing_scratch("avgflx_clock")
    for axis, (slot, flux) in enumerate(zip(HISTORY, fluxes)):
        out = state.existing_scratch(slot)
        get_kernel("avgflx", "update_avgflx")(
            ((out.size + 255) // 256,), (256,),
            (out, flux, clock, np.int64(out.size),
             np.int32(cfg.nx), np.int32(cfg.ny), np.int32(axis)))
    n = cfg.nx * cfg.ny
    get_kernel("avgflx", "advance_avgflx_count")(
        ((n + 255) // 256,), (256,), (clock, np.int32(n)))


def history_written(state):
    """Arm next-step reset, preserving the just-written mean and count."""
    from woof.core.streaming import live_scratch

    streamed = getattr(state, "_streamed_domain", None)
    arm = getattr(streamed, "arm_avgflx_reset", None)
    if arm is not None and arm():
        clock = state.existing_scratch("avgflx_clock")
        if clock is not None:
            clock[1, ...] = 1
        return
    for clock in live_scratch(state, "avgflx_clock"):
        clock[1, ...] = 1
