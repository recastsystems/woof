"""WRF KF-CuP's radiation-driver override, on private radiation buffers.

Source: WRF v4.6.1 phys/module_radiation_driver.F:1510-1557.
The held CuP cloud is updated by the native clear rule; prognostic QC is not.
"""
from __future__ import annotations

import numpy as np


def merge_kfcup_clouds(qc, fraction, flags, fields, cfg, *, level_major=False):
    import cupy as cp
    from woof.core.kernels import get_kernel

    nz, ncol = qc.shape if level_major else (qc.shape[1], qc.shape[0])
    pack = lambda a: cp.ascontiguousarray(a if level_major else a.T)
    liquid, cf = pack(qc), pack(fraction)
    cup = cp.ascontiguousarray(fields['cldfra_cup'].reshape(nz, ncol))
    flag = (cp.zeros((nz, ncol), dtype=cp.int32) if flags is None else pack(flags))
    surface = tuple(cp.ascontiguousarray(fields[name].reshape(ncol))
                    for name in ('shall', 'cubot', 'cutop'))
    get_kernel('kfcup', 'kfcup_radiation_merge')(
        ((liquid.size + 255)//256,), (256,),
        (cf, liquid, cup, flag, *surface, np.int32(cfg.icloud),
         np.int32(cfg.shallowcu_forced_ra), np.int32(ncol), np.int32(nz)))
    fields['cldfra_cup'][...] = cup.reshape(fields['cldfra_cup'].shape)
    return (liquid, cf) if level_major else (liquid.T.copy(), cf.T.copy())
