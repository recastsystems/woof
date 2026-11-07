"""Scalar PBL-driver words for optional column schemes."""
import numpy as _np
from woof.core.constants import RD, RV, G

def pbl_driver_constants_f32() -> tuple:
    """WRF's PBL-driver scalar words as BL_GFS-style wrappers receive them.

    module_pbl_driver.F passes CP=cp, G=g, ROVCP=rcp, R=r_d, ROVG=rovg,
    EP1=ep_1 and KARMAN=karman (e.g. :1414-1419 for the GFS scheme): all
    default-REAL PARAMETERs of module_model_constants.F, which gfortran
    folds in binary32 one operation at a time (this module's header), so
    each is spelled as that chain.  Returned in that argument order as
    numpy float32 scalars.
    """
    f = _np.float32
    r_d, r_v, g = f(RD), f(RV), f(G)
    cp_ = f(f(f(7.0) * r_d) / f(2.0))
    return (cp_, g, f(r_d / cp_), r_d, f(r_d / g), f(f(r_v / r_d) - f(1.0)),
            f(0.4))
