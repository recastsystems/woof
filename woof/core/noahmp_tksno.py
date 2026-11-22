"""Snow thermal conductivity selector ``opt_tksno``, host reference.

Re-implemented from published descriptions, not bitwise to upstream. No
Noah-MP v5 source (NCAR/noahmp, the WRF 4.8 noahmp submodule, or any
UCAR-licensed copy) was opened, read or quoted. Noah-MP v5 added a namelist
choice for the snow thermal conductivity that v4 hard-codes (He et al. 2023,
GMD 16:5131, Sect. 2.6; option list in the public NCAR Noah-MP tutorial, He,
25 May 2023). Each formula below is taken from its own paper:

====  ==============================  ===================================
opt   source                          k(rho), W m-1 K-1, rho in kg m-3
====  ==============================  ===================================
1     Yen (1965), Stieglitz form      3.2217e-6 rho^2  (the v4 line; default)
2     Anderson (1976), NWS-19         0.02 + 2.5e-6 rho^2
3     constant                        0.35
4     Verseghy (1991), CLASS I        2.576e-6 rho^2 + 0.074
5     Yen (1981); Douville (1995)     2.22 (rho/1000)^1.88
====  ==============================  ===================================

``rho`` is CSNOW's bulk layer density (SNICE + SNLIQ) / DZSNSO. The device
selector is ``noahmp_tksno_selected`` in
``woof/core/kernels/noahmp_tksno_options.cuh``; this module is the float32
host reference its unit test checks against, and the admission check every
door shares. Full provenance: ``docs/noahmp-opt-tksno.md``.
"""
from __future__ import annotations

import numpy as np

#: Admitted values; 1 is the default and the v4 formula.
OPT_TKSNO_VALUES = (1, 2, 3, 4, 5)

#: The label every selected run carries.
LABEL = "re-implemented from published descriptions, not bitwise to upstream"

_F = np.float32


def validate(opt_tksno) -> None:
    """Refuse a value no formula exists for (nothing would compute k)."""
    if type(opt_tksno) is not int or opt_tksno not in OPT_TKSNO_VALUES:
        raise ValueError(
            f"opt_tksno must be an int in {OPT_TKSNO_VALUES}, got {opt_tksno!r}")


def snow_thermal_conductivity(rho, opt_tksno: int = 1):
    """k(rho) in float32, one operation at a time like the device code."""
    validate(opt_tksno)
    rho = np.asarray(rho, dtype=_F)
    sq = (rho * rho).astype(_F)
    if opt_tksno == 1:
        return (_F(3.2217e-6) * sq).astype(_F)
    if opt_tksno == 2:
        return (_F(0.02) + (_F(2.5e-6) * sq).astype(_F)).astype(_F)
    if opt_tksno == 3:
        return np.full_like(rho, _F(0.35))
    if opt_tksno == 4:
        return ((_F(2.576e-6) * sq).astype(_F) + _F(0.074)).astype(_F)
    return (_F(2.22) * np.power((rho / _F(1000.0)).astype(_F), _F(1.88)).astype(_F)).astype(_F)


__all__ = ["LABEL", "OPT_TKSNO_VALUES", "snow_thermal_conductivity", "validate"]
