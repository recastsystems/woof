"""CUDA launch and calendar orchestration for WRF's TMNUPDATE.

All field arithmetic is in the shared-loader kernel. Both arithmetic modes
use that kernel and explicit rounded operations. Counters live on the device
in the driver's field inventory, so the restart writer carries them in place.
"""
from datetime import timedelta
import numpy as np
from woof.core.kernels import load_module
from woof.wrf_namelist_registry import TMN_UPDATE_STATE


def initialize(fields, lagday):
    import cupy as cp
    tmn = fields["tmn"]
    fields["tlag"] = cp.repeat(tmn[None], lagday, axis=0)
    fields["tyr"] = tmn.copy()
    fields["tyra"] = cp.zeros_like(tmn)
    fields["tdly"] = cp.zeros_like(tmn)
    fields["nday"] = cp.zeros(1, dtype=cp.float32)
    fields["nyear"] = cp.ones(1, dtype=cp.int32)


def launch(fields, tsk, *, julian, dt, year, module=None):
    import cupy as cp
    shape = tuple(tsk.shape)
    if len(shape) != 2 or tsk.dtype != cp.float32 or not tsk.flags.c_contiguous:
        raise ValueError("TMN update requires contiguous float32 TSK (y,x)")
    lagday = fields["tlag"].shape[0]
    for key, row in TMN_UPDATE_STATE.items():
        value = fields[key]
        expected = ((lagday, *shape) if key == "tlag" else
                    (1,) if key in ("nday", "nyear") else shape)
        dtype = cp.int32 if row["dtype"] == "int32" else cp.float32
        if (tuple(value.shape) != expected or value.dtype != dtype
                or not value.flags.c_contiguous or value.device.id != tsk.device.id):
            raise ValueError(f"TMN update {key} must be contiguous {dtype} {expected} on the TSK device")
    mod = load_module("tmn_update") if module is None else module
    args = (np.float32(julian), np.float32(dt), np.int32(year))
    ncol = tsk.size
    mod.get_function("tmn_columns")(((ncol+127)//128,), (128,),
        (tsk, fields["tmn"], fields["tlag"], fields["tyr"], fields["tyra"],
         fields["tdly"], fields["nday"], fields["nyear"], np.int32(ncol),
         np.int32(lagday), *args))
    mod.get_function("tmn_counters")((1,), (1,),
        (fields["nday"], fields["nyear"], *args))


def advance(fields, *, start_time, elapsed_seconds, dt):
    if start_time is None:
        raise ValueError("tmn_update=1 requires the forecast start time: without it daily and yearly means update at the wrong calendar boundary")
    valid = start_time + timedelta(seconds=float(elapsed_seconds))
    hour = (valid.hour + valid.minute/60. + valid.second/3600.
            + valid.microsecond/3.6e9)
    # module_domain domain_clock_get supplies one-based fractional day.
    julian = np.float32(valid.timetuple().tm_yday + hour/24.)
    launch(fields, fields["tsk"], julian=julian, dt=dt, year=valid.year)
