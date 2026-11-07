"""WRF 4.7.1 ideal SCM prescribed-flux surface, selector 89.

The six forcing controls are native scalar state, not namelist options.
Advance each once per surface call, then fill every column on the card.
No diagnostic repair is applied to native unassigned near-surface outputs.
"""
from functools import lru_cache

import numpy as np

from woof.core.physics_inventory import SCM_SURFACE_FORCING_FIELDS

MODULE_KEY = "woof.core.idealscmsfclay:idealscmsfclay"
OUTPUTS = ("hfx", "lh", "qfx", "tsk", "flhc", "exch_temf", "flqc")


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from woof.core.kernels import compile_noftz_module, module_source
    with cp.cuda.Device(device):
        return compile_noftz_module("idealscmsfclay", module_source("idealscmsfclay"), MODULE_KEY)


def frame_module():
    """The actual production image and its source for frame measurements."""
    import cupy as cp
    from woof.core.kernels import module_source
    return _module(cp.cuda.Device().id), module_source("idealscmsfclay")


def launch(th, pi, rho, mavail, forcing, outputs, *, dt, cp_air=1004.5, xlv=2.5e6):
    """Run IDEALSCMSFCLAY, mutating forcing scalars and seven outputs."""
    import cupy as cp
    shape = th.shape
    for value in (th, pi, rho, mavail, *(outputs[n] for n in OUTPUTS)):
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                or value.shape != shape or not value.flags.c_contiguous):
            raise TypeError("ideal SCM requires contiguous float32 surface arrays")
    for name in SCM_SURFACE_FORCING_FIELDS:
        value = forcing[name]
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                or value.size != 1 or not value.flags.c_contiguous):
            raise TypeError(f"ideal SCM {name} must be a float32 scalar carrier")
    mod = _module(cp.cuda.Device().id)
    mod.get_function("ideal_scm_advance")((1,), (1,),
        (*(forcing[n] for n in SCM_SURFACE_FORCING_FIELDS), np.float32(dt)))
    n = th.size
    if n:
        mod.get_function("ideal_scm_surface")(((n + 127) // 128,), (128,),
            (np.int32(n), th, pi, rho, mavail,
             *(forcing[n] for n in SCM_SURFACE_FORCING_FIELDS[:3]),
             np.float32(cp_air), np.float32(xlv), *(outputs[n] for n in OUTPUTS)))
