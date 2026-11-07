"""NOAA-EMC/HRRR 40ee6058 module_diag_cl.F, on the device.

The arrays themselves are sums/sums of squares until the closing step,
when the fork replaces them with means/population standard deviations.
History reads never finalize or reset them. Window units are minutes,
despite the old routine's comments saying hours.
"""
from __future__ import annotations

import numpy as np

HISTORY = {}
for stem, units, description in (
    ("t2", "K", "2 m temperature"),
    ("q2", "kg kg-1", "2 m water vapor mixing ratio"),
    ("skintemp", "K", "Skin temperature"),
    ("swrad", "W m-2", "Downward shortwave radiation"),
    ("swnorm", "W m-2", "Direct normal shortwave radiation"),
):
    for suffix in ("min", "max"):
        slot = stem + suffix
        HISTORY[slot] = (slot.upper(), units, description + " interval " + suffix)
    for suffix in ("min", "max"):
        slot = "t" + stem + suffix
        HISTORY[slot] = (slot.upper(), "minute", "Time of " + description.lower() + " " + suffix)
    for suffix in ("mean", "std"):
        slot = stem + suffix
        HISTORY[slot] = (slot.upper(), units, description + " interval " + suffix)
for slot in ("u10max", "v10max", "spduv10max", "tspduv10max",
             "u10mean", "v10mean", "spduv10mean", "u10std", "v10std", "spduv10std"):
    HISTORY[slot] = (slot.upper(), "minute" if slot.startswith("t") else "m s-1",
                     "Climate interval wind statistic " + slot)
for stem in ("raincv", "rainncv"):
    for slot in (stem + "max", "t" + stem + "max", stem + "mean", stem + "std"):
        HISTORY[slot] = (slot.upper(), "minute" if slot.startswith("t") else "kg m-2 s-1",
                         "Climate interval precipitation flux statistic " + slot)

# Counter values are overwritten by INT(window*60/dt), not incremented.
# Planes keep their ownership identical on resident and streamed routes.
PERSISTENT = (*HISTORY, "cl_nsteps", "cl_nsteps_wind")


def scratch_shapes(cfg):
    if not cfg.output_diagnostics or cfg.auxhist3_interval:
        return {}
    return {slot: (cfg.ny, cfg.nx) for slot in (*PERSISTENT, "cl_zero")}


def update(state, cfg, *, is_restart=False, curr_secs2=None):
    """Sample the completed physics step, before advancing its clock.

    ``is_restart`` and ``curr_secs2`` expose the native routine's restart
    call semantics for replay. WOOF checkpoints restore these arrays and
    the absolute clock and continue the same window, without restarting it.
    """
    if not cfg.output_diagnostics:
        return
    if cfg.auxhist3_interval:
        from woof.core.clwrf import update as stock_update
        stock_update(state, cfg, curr_secs2=curr_secs2)
        return
    from woof.core.kernels import get_kernel

    zero = state.existing_scratch("cl_zero")
    fields = getattr(state.physics, "fields", {})
    surface = tuple(fields.get(name, zero) for name in
                    ("t2", "q2", "tsk", "swdown", "swddni", "u10", "v10"))
    micro = getattr(state.physics, "microphysics", None)
    rainc = getattr(state.physics, "cu_raincv", None)
    rain = getattr(micro, "rainncv", None) if cfg.mp_physics else None
    get_kernel("climate_diagnostics", "clwrf_output_calc")(
        (((cfg.nx * cfg.ny + 255) // 256),), (256,),
        (*surface, zero if rainc is None else rainc, zero if rain is None else rain,
         *(state.existing_scratch(slot) for slot in PERSISTENT),
         np.int32(cfg.nx * cfg.ny), np.float32(cfg.dt),
         np.float32(state.elapsed_seconds / 60.),
         np.float32(state.elapsed_seconds if curr_secs2 is None else curr_secs2),
         np.int32(cfg.diag_int), np.int32(cfg.wind_int), np.int32(is_restart)))
