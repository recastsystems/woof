"""Scale-aware ground snow cover fraction for Noah-MP (``opt_scf = 2``).

Re-implemented from published descriptions, not bitwise to upstream.

Source: Abolafia-Rosenzweig, R., C. He, T.-S. Lin, M. Barlage and K. Rittger,
2025: Improved cross-scale snow cover simulations by developing a scale-aware
ground snow cover fraction parameterization in the Noah-MP land surface model.
J. Adv. Model. Earth Syst., 17(6), e2024MS004704, doi:10.1029/2024MS004704.
CC BY 4.0 (NOAA Institutional Repository, noaa/71110).

Only the paper was read.  No Noah-MP v5 source file was opened.

What the paper keeps (its Section 2.2, Eq. 1-3, the Niu and Yang 2007 form our
WRF-lineage port already computes in ENERGY):

    SCF   = tanh(h_sno / (SCfactor * f_melt))                       (Eq. 1)
    f_melt = (rho_snow / rho_new) ** m,  rho_new = 100 kg m-3         (Eq. 2)
    rho_snow = SWE / h_sno                                           (Eq. 3)

What the paper changes (Section 2.3, Eq. 4-5): SCfactor and m stop being
land-cover table values (MPTABLE SCFFAC and MFSNO) and become functions of the
model grid size DX in km, the same for every land type:

    SCfactor = b1 * sinh(b2 * DX) + b2,  [b1, b2] = [0.0062, 0.0555]  (Eq. 4)
    m        = c1 + tanh(c2 * DX),        [c1, c2] = [0.9713, 0.7436]  (Eq. 5)

and Section 2.3 bounds both above by their DX = 36 km solutions (the coarsest
grid the fit used; 0.0779 and 1.9713 as printed).  Both functions increase
with DX, so the bound is applied by evaluating them at min(DX, 36 km): that is
the paper's "solutions of Equations 4 and 5 when DX = 36-km" exactly, rather
than its four-digit rounding.

DX is the grid spacing of the column: the domain ``dx`` on WOOF-ArWen, the local
cell spacing on a hex variable-resolution mesh (``noahmp_cell_dx`` field).

Evaluated in float64 and rounded once to float32, so the two parameters are
the correctly rounded binary32 values of the paper's formulas for nearly every
DX.  The SCF statement itself is unchanged: the same glibc ``tanhf`` and
``powf`` transcriptions the default path uses.
"""

from __future__ import annotations

import numpy as np

LABEL = "re-implemented from published descriptions, not bitwise to upstream"
CITATION = ("Abolafia-Rosenzweig et al. 2025, JAMES 17(6), e2024MS004704, "
            "doi:10.1029/2024MS004704 (CC BY 4.0), Section 2.3, Eq. 4-5")

#: Eq. 4 coefficients [beta1_fsno, beta2_fsno].
SCFACTOR_COEFFICIENTS = (0.0062, 0.0555)
#: Eq. 5 coefficients [beta1_m, beta2_m].
MELT_EXPONENT_COEFFICIENTS = (0.9713, 0.7436)
#: Section 2.3 upper constraint: the coarsest grid of the fit, km.
DX_CAP_KM = 36.0


REFUSAL = ("opt_scf=2 needs a positive finite grid spacing in metres for "
           "every land column")


def check_spacing(dx_m) -> None:
    """Refuse a non-positive or non-finite spacing.

    DX is a physical grid size, and a zero would silently give the DX -> 0
    limit.  On a device array this reads one flag back to the host, so the
    land step calls it once per spacing source, never per step.
    """
    try:
        import cupy as cp
        xp = cp.get_array_module(dx_m)
    except ImportError:
        xp = np
    dx = xp.asarray(dx_m, dtype=xp.float64)
    if bool((~(xp.isfinite(dx) & (dx > 0.0))).any()):
        raise ValueError(REFUSAL)


def scale_aware_parameters(dx_m, *, check=True):
    """``(SCfactor, m)`` as float32 arrays for grid spacing ``dx_m`` in metres.

    Works on NumPy and CuPy arrays (or a scalar).  ``check=False`` is for the
    per-step device path, whose spacing the land step has already checked
    (:func:`check_spacing`): it launches no host read-back.
    """
    try:
        import cupy as cp
        xp = cp.get_array_module(dx_m)
    except ImportError:  # host-only use (tests, documentation)
        xp = np
    if check:
        check_spacing(dx_m)
    dx = xp.asarray(dx_m, dtype=xp.float64)
    dx_km = xp.minimum(dx / 1000.0, DX_CAP_KM)
    b1, b2 = SCFACTOR_COEFFICIENTS
    c1, c2 = MELT_EXPONENT_COEFFICIENTS
    scfactor = (b1 * xp.sinh(b2 * dx_km) + b2).astype(xp.float32)
    melt_exponent = (c1 + xp.tanh(c2 * dx_km)).astype(xp.float32)
    return scfactor, melt_exponent


def ground_snow_cover_fraction(snowh, sneqv, dx_m):
    """Host float64 reference of Eq. 1-5, for tests and hand checks only."""
    snowh = np.asarray(snowh, dtype=np.float64)
    sneqv = np.asarray(sneqv, dtype=np.float64)
    dx_km = np.minimum(np.asarray(dx_m, dtype=np.float64) / 1000.0, DX_CAP_KM)
    b1, b2 = SCFACTOR_COEFFICIENTS
    c1, c2 = MELT_EXPONENT_COEFFICIENTS
    scfactor = b1 * np.sinh(b2 * dx_km) + b2
    m = c1 + np.tanh(c2 * dx_km)
    with np.errstate(divide="ignore", invalid="ignore"):
        fmelt = (sneqv / np.where(snowh > 0, snowh, 1.0) / 100.0) ** m
        scf = np.tanh(snowh / (scfactor * fmelt))
    return np.where(snowh > 0, scf, 0.0)
