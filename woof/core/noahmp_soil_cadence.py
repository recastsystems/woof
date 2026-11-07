"""WRF v4.6.1 soil cadence (module_sf_noahmpdrv.F:649-678).

Source is WRF public-domain Noah-MP v4, never the v5/MPAS refactor.
The absolute driver timestep and the nine Registry accumulator arrays are
restart state. No device value is read back to decide the cadence.
"""
from __future__ import annotations
import math
import numpy as np

# Formal SFLX name -> native Registry carrier. ETRANI has four soil layers.
ACCUMULATORS = {
    "acc_ssoil":"acc_ssoil", "acc_qinsur":"acc_qinsur",
    "acc_qseva":"acc_qseva", "acc_etrani":"acc_etrani",
    "acc_dwater":"acc_dwaterxy", "acc_prcp":"acc_prcpxy",
    "acc_ecan":"acc_ecanxy", "acc_etran":"acc_etranxy",
    "acc_edir":"acc_edirxy",
}

def validate(value):
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError("soiltstep must be a finite nonnegative float in seconds")
    return value

def shapes(value, shape):
    return {name:((4,*shape) if formal=="acc_etrani" else tuple(shape))
            for formal,name in ACCUMULATORS.items()} if validate(float(value))>0 else {}

def cadence(value, dt, itimestep):
    validate(float(value))
    # NINT of WRF's REAL(binary32) division, with positive half ties away
    # from zero. Python's round would use ties-to-even and is not native.
    ratio=np.float32(np.float32(value)/np.float32(dt))
    if not np.isfinite(ratio) or ratio > np.iinfo(np.int32).max-1:
        raise ValueError("soiltstep/DT exceeds the native INTEGER cadence")
    steps=max(int(math.floor(float(ratio)+0.5)),1)
    step=int(itimestep)
    return steps,step%steps==0,steps==1 or step%steps==1
