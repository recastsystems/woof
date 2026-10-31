"""RRFS Noah land and its required post-LSM surface diagnostics.

The Noah case in mpas_atmphys_sfc_diagnostics.F is unchanged from v8.4.2
at fork 1d6e370e. Register it on the land row so every surface scheme
receives the same driver diagnostics after Noah and sea ice complete.
"""
from ..noah_v842 import bind_native_pools, run_slot
from ..sfclay_mm5_v842 import run_native_diagnostics
