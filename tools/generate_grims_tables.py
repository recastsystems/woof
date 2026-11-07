"""Generate the packaged GRIMS shallow-convection lookup tables.

WRF v4.6.1 ``phys/module_shcu_grims.F`` builds five tables once, in
``grimsinit`` (:940-976), and every column call interpolates them:

========  ============  =====================================================
table     shape         what it holds (source lines)
========  ============  =====================================================
tbtdp     5001          dewpoint temperature of a vapour pressure, by Newton
                        iteration warm-started from the previous entry
                        (funct_dew_point_temp_init :980-1001, ftdpxg :1100)
tbthe     151 x 241     equivalent potential temperature on (T, p**kappa)
                        (funct_pot_temp_init :1005-1035, fthex :1132)
tbtma     121 x 151     moist-adiabat parcel temperature on (theta_e,
tbqma                   p**kappa) and its specific humidity, Newton warm-
                        started along each row (funct_moist_adiabat_init
                        :1039-1072, ftmaxg :1159)
tbpvs     7501          saturation vapour pressure over water/ice
                        (funct_svp_init :1076-1096, fpvsx :1200)
========  ============  =====================================================

plus the twelve interpolation constants c1x*/c2x*/c1y*/c2y* the same
routines store in module SAVE variables.  2-D tables are stored row-major
as ``[jy, jx]``, which is Fortran's ``tb(jx, jy)`` memory order.

Precision and elementary functions
----------------------------------
Every variable is default ``REAL`` and WRF does not promote it, so this is
binary32 throughout.  The module PARAMETERs (``rocp = rd/cp`` ...) are
constant expressions gfortran folds at compile time, one correctly rounded
binary32 operation each, which is what a ``np.float32`` operation is.  The
two exponentiations of LITERALS by a parameter, ``0.04**rocp``,
``0.01**rocp`` and ``1.10**rocp``, are folded too (correctly rounded); every
other ``**`` with a real exponent and every ``exp`` is a run-time ``powf``/
``expf`` call (``nm -u`` on WRF's own object: expf, powf), and here they are
WOOF's own float32 routines in :mod:`woof.core.noahmp_libm`.  ``t**2`` is
an integer power, ``t*t``.

The output is checked word for word against the tables WRF's own
``grimsinit`` built, dumped by ``tools/grims_wrf461_oracle``
(``tests/test_grims_tables.py``).

Usage: ``python tools/generate_grims_tables.py`` rewrites
``woof/data/grims/grims_tables.npz``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from woof.core.noahmp_libm import expf, powf

F = np.float32

NXTDP = 5001
NXTHE, NYTHE = 241, 151
NXMA, NYMA = 151, 121
NXSVP = 7501

# module_shcu_grims.F:13-32, folded exactly as gfortran folds them.
T0C = F(2.7315e+2)
PSAT = F(6.1078e+2)
RD = F(2.8705e+2)
RV = F(4.6150e+2)
CP = F(1.0046e+3)
HVAP = F(2.5000e+6)
CVAP = F(1.8460e+3)
CLIQ = F(4.1855e+3)
CICE = F(2.1060e+3)
HSUB = F(2.8340e+6)
TERRM = F(1.0e-4)
ROCP = F(RD / CP)
CPOR = F(CP / RD)
EPS = F(RD / RV)
TTP = F(T0C + F(0.01))
PSATK = F(PSAT * F(1.0e-3))
PSATB = F(PSATK * F(1.0e-2))
DLDT = F(CVAP - CLIQ)
XA = F(-F(DLDT / RV))
XB = F(XA + F(HVAP / F(RV * TTP)))
DLDTI = F(CVAP - CICE)
XAI = F(-F(DLDTI / RV))
XBI = F(XAI + F(HSUB / F(RV * TTP)))


def folded_power(base: float, exponent: np.float32) -> np.float32:
    """A literal ``REAL`` raised to a ``REAL`` parameter, as gfortran folds it.

    gfortran evaluates a constant ``a**b`` with MPFR and rounds once to
    binary32.  binary64 ``pow`` of the two binary32 operands, rounded once
    more, is the same word unless the exact result lies within 2**-29
    relative of a binary32 midpoint; the oracle comparison in
    ``tests/test_grims_tables.py`` covers the three values used here.
    """
    return F(math.pow(float(F(base)), float(exponent)))


def _psat(scale: np.float32, tr: np.float32, xa: np.float32,
          xb: np.float32) -> np.float32:
    """``scale*(tr**xa)*exp(xb*(1.-tr))``, left to right as WRF writes it."""
    return F(F(scale * powf(tr, xa)) * expf(F(xb * F(F(1.0) - tr))))


def ftdpxg(tg: np.float32, pv: np.float32) -> np.float32:
    """module_shcu_grims.F:1100-1128."""
    t = F(tg)
    tr = F(TTP / t)
    pvt = _psat(PSATK, tr, XA, XB)
    el = F(HVAP + F(DLDT * F(t - TTP)))
    dpvt = F(F(el * pvt) / F(RV * F(t * t)))
    terr = F(F(pvt - pv) / dpvt)
    t = F(t - terr)
    while abs(terr) > TERRM:
        tr = F(TTP / t)
        pvt = _psat(PSATK, tr, XA, XB)
        el = F(HVAP + F(DLDT * F(t - TTP)))
        dpvt = F(F(el * pvt) / F(RV * F(t * t)))
        terr = F(F(pvt - pv) / dpvt)
        t = F(t - terr)
    return t


def fthex(t: np.float32, pk: np.float32) -> np.float32:
    """module_shcu_grims.F:1132-1155."""
    p = powf(pk, CPOR)
    tr = F(TTP / t)
    pv = _psat(PSATB, tr, XA, XB)
    pd = F(p - pv)
    if pd > F(0.0):
        el = F(HVAP + F(DLDT * F(t - TTP)))
        expo = F(F(F(el * EPS) * pv) / F(F(CP * t) * pd))
        expo = min(expo, F(100.0))
        return F(F(t * powf(pd, F(-ROCP))) * expf(expo))
    return F(0.0)


def ftmaxg(tg: np.float32, the: np.float32,
           pk: np.float32) -> tuple[np.float32, np.float32]:
    """module_shcu_grims.F:1159-1196; returns (t, qma)."""

    def step(t, p):
        tr = F(TTP / t)
        pv = _psat(PSATB, tr, XA, XB)
        pd = F(p - pv)
        el = F(HVAP + F(DLDT * F(t - TTP)))
        expo = F(F(F(el * EPS) * pv) / F(F(CP * t) * pd))
        thet = F(F(t * powf(pd, F(-ROCP))) * expf(expo))
        inner = F(F(F(DLDT * t) / el) + F(F(el * p) / F(F(RV * t) * pd)))
        dthet = F(F(thet / t) * F(F(1.0) + F(expo * inner)))
        terr = F(F(thet - the) / dthet)
        return F(t - terr), terr

    t = F(tg)
    p = powf(pk, CPOR)
    t, terr = step(t, p)
    while abs(terr) > TERRM:
        t, terr = step(t, p)
    tr = F(TTP / t)
    pv = _psat(PSATB, tr, XA, XB)
    pd = F(p - pv)
    qma = F(F(EPS * pv) / F(pd + F(EPS * pv)))
    return t, qma


def fpvsx(t: np.float32) -> np.float32:
    """module_shcu_grims.F:1200-1217."""
    tr = F(TTP / t)
    if t >= TTP:
        return _psat(PSATK, tr, XA, XB)
    return _psat(PSATK, tr, XAI, XBI)


def _axis(xmin: np.float32, xmax: np.float32, n: int):
    """``xinc``, ``c1x``, ``c2x`` exactly as each init routine forms them."""
    xinc = F(F(xmax - xmin) / F(n - 1))
    return xinc, F(F(1.0) - F(xmin / xinc)), F(F(1.0) / xinc)


def generate() -> dict[str, np.ndarray]:
    """Every grimsinit table and constant, in WRF's construction order.

    The theta_e table overflows to +Inf at its hot, low-pressure corner,
    exactly as WRF's does (``t*pd**(-rocp)*exp(expo)`` with ``expo``
    capped at 100); the overflow is the table's content, not an error.
    """
    with np.errstate(over="ignore"):
        return _generate()


def _generate() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    # funct_dew_point_temp_init
    xmin, xmax = F(0.001), F(10.001)
    xinc, c1xtdp, c2xtdp = _axis(xmin, xmax, NXTDP)
    tbtdp = np.empty(NXTDP, F)
    t = F(208.0)
    for jx in range(1, NXTDP + 1):
        x = F(xmin + F(F(jx - 1) * xinc))
        t = ftdpxg(t, x)
        tbtdp[jx - 1] = t

    # funct_pot_temp_init
    xmin, xmax = F(TTP - F(90.0)), F(TTP + F(30.0))
    xinc, c1xthe, c2xthe = _axis(xmin, xmax, NXTHE)
    ymin, ymax = folded_power(0.04, ROCP), folded_power(1.10, ROCP)
    yinc, c1ythe, c2ythe = _axis(ymin, ymax, NYTHE)
    tbthe = np.empty((NYTHE, NXTHE), F)
    for jy in range(1, NYTHE + 1):
        pk = F(ymin + F(F(jy - 1) * yinc))
        for jx in range(1, NXTHE + 1):
            tbthe[jy - 1, jx - 1] = fthex(F(xmin + F(F(jx - 1) * xinc)), pk)

    # funct_moist_adiabat_init
    xmin, xmax = F(200.0), F(500.0)
    xinc, c1xma, c2xma = _axis(xmin, xmax, NXMA)
    ymin, ymax = folded_power(0.01, ROCP), folded_power(1.10, ROCP)
    yinc, c1yma, c2yma = _axis(ymin, ymax, NYMA)
    tbtma = np.empty((NYMA, NXMA), F)
    tbqma = np.empty((NYMA, NXMA), F)
    for jy in range(1, NYMA + 1):
        pk = F(ymin + F(F(jy - 1) * yinc))
        t = F(xmin * pk)
        for jx in range(1, NXMA + 1):
            the = F(xmin + F(F(jx - 1) * xinc))
            t, q = ftmaxg(t, the, pk)
            tbtma[jy - 1, jx - 1] = t
            tbqma[jy - 1, jx - 1] = q

    # funct_svp_init
    xmin, xmax = F(180.0), F(330.0)
    xinc, c1xpvs, c2xpvs = _axis(xmin, xmax, NXSVP)
    tbpvs = np.empty(NXSVP, F)
    for jx in range(1, NXSVP + 1):
        tbpvs[jx - 1] = fpvsx(F(xmin + F(F(jx - 1) * xinc)))

    out.update(tbtdp=tbtdp, tbthe=tbthe, tbtma=tbtma, tbqma=tbqma,
               tbpvs=tbpvs)
    # The same order run_grims.F90 dumps them in.
    out["constants"] = np.array(
        [c1xma, c2xma, c1yma, c2yma, c1xpvs, c2xpvs, c1xtdp, c2xtdp,
         c1xthe, c2xthe, c1ythe, c2ythe], F)
    return out


#: Order and names of ``constants``.
CONSTANT_NAMES = ("c1xma", "c2xma", "c1yma", "c2yma", "c1xpvs", "c2xpvs",
                  "c1xtdp", "c2xtdp", "c1xthe", "c2xthe", "c1ythe", "c2ythe")


def write(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed: savez_compressed embeds zlib output, whose bytes are a
    # property of the zlib build, and the file is pinned by its SHA-256.
    tables = generate()
    with open(destination, "wb") as handle:
        np.savez(handle, **tables)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    write(root / "woof" / "data" / "grims" / "grims_tables.npz")
