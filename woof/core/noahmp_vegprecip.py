"""Bitwise FP32 transcriptions of the WRF v4.6.1 Noah-MP vegetation-phenology
and precipitation-heat leaves.

Source of truth
---------------
``phys/module_sf_noahmplsm.F`` of the pinned WRF checkout

    tree   <wrf-4.6.1-checkout>
    commit d66e442fccc04111067e29274c9f9eaccc3cef28
    sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282

    SUBROUTINE PHENOLOGY   lines 1255-1358
    SUBROUTINE PRECIP_HEAT lines 1362-1556

``kind_phys == kind(1.0)`` in that build, so every declared ``REAL`` is
binary32.  Each arithmetic result below is rounded to binary32 exactly once,
by :func:`woof.core.noahmp_libm.f32`.  Rounding a binary64 intermediate to
binary32 is safe for ``+ - * /`` and ``sqrt`` because binary64 carries
53 >= 2*24 + 2 bits, so the double rounding is provably innocuous; it is *not*
safe across a fused expression, which is why no expression here spans more
than one operation.

Transcendentals
---------------
gfortran lowers ``EXP`` on a ``REAL(4)`` to a call to glibc ``expf`` and
``x ** 0.667`` to a call to glibc ``powf``.  Neither is correctly rounded, so
``np.exp``/``**`` cannot hold a max_ulp-0 gate.  Both come from
:mod:`woof.core.noahmp_libm`, which reproduces glibc 2.39's own kernels.

``MOD`` on ``REAL(4)`` is lowered to x87 ``fprem``, which is exact, so
``math.fmod`` on the binary64 promotions reproduces it bit for bit.

MIN / MAX
---------
gfortran emits ``minss``/``maxss`` with the first argument in the destination,
so ``MAX(a,b)`` is ``a > b ? a : b`` and ``MIN(a,b)`` is ``a < b ? a : b``:
both return the *second* argument on a tie.  That is observable -- it decides
the sign of a zero result -- and is reproduced by :func:`_fmax` / :func:`_fmin`
rather than by Python's ``max``/``min``, which return the first.

Pinned option identity
----------------------
The WRF Registry defaults, as asserted by the oracle driver:

* ``dveg = 4``   -- table LAI/SAI with maximum vegetation fraction.  The
  selected DVEG7/8/9 branch uses the actual carried LAI and derives SAI
  with the native threshold and burial rules. Dynamic carbon vegetation
  remains outside this prescribed-phenology reference.
* ``opt_crop = 0`` -- ``module_sf_noahmpdrv.F`` line 767 sets ``CROPTYPE = 0``
  unconditionally and only revises it under ``IOPT_CROP > 0``, so ``croptype``
  is identically zero.  Every ``CROPTYPE > 0`` disjunct is therefore dead, and
  with it the only use of ``PGS``.  :func:`phenology` refuses a nonzero
  ``croptype``.

Neither leaf reads any other option variable.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

from woof.core.noahmp_libm import expf, f32, powf

__all__ = [
    "PhenologyResult",
    "PrecipHeatResult",
    "phenology",
    "precip_heat",
    "CWAT",
    "CICE",
    "TFRZ",
]

# module_sf_noahmplsm.F lines 207-212.  Both quotients below are folded by the
# compiler; 4188000 and 2094000 are < 2**24 so both fold exactly.
CWAT = f32(4.188e06)
CICE = f32(2.094e06)
TFRZ = f32(273.16)

_CWAT_PER_1000 = f32(CWAT / 1000.0)  # 4188.0, exact
_CICE_PER_1000 = f32(CICE / 1000.0)  # 2094.0, exact


def _fmax(a: float, b: float) -> float:
    """gfortran ``MAX(a, b)``: ``maxss`` returns the second operand on a tie."""
    return a if a > b else b


def _fmin(a: float, b: float) -> float:
    """gfortran ``MIN(a, b)``: ``minss`` returns the second operand on a tie."""
    return a if a < b else b


# ==========================================================================
# PHENOLOGY
# ==========================================================================
class PhenologyResult(NamedTuple):
    lai: float   # INTENT(INOUT), as it comes back
    sai: float   # INTENT(INOUT), as it comes back
    elai: float
    esai: float
    igs: float
    fb: float


def phenology(
    *,
    dveg: int,
    vegtyp: int,
    croptype: int,
    snowh: float,
    tv: float,
    lat: float,
    yearlen: int,
    julian: float,
    lai: float,
    sai: float,
    troot: float,
    pgs: int,
    iswater: int,
    isbarren: int,
    isice: int,
    urban_flag: bool,
    hvt: float,
    hvb: float,
    tmin: float,
    laim: Sequence[float],
    saim: Sequence[float],
) -> PhenologyResult:
    """PHENOLOGY, module_sf_noahmplsm.F:1255.

    ``troot`` and ``pgs`` are accepted because the WRF argument list carries
    them.  ``troot`` appears nowhere in the routine body at any option setting;
    ``pgs`` is read only inside a ``CROPTYPE > 0`` disjunct, which ``opt_crop=0``
    makes unreachable.  Both are therefore dead and are asserted, not used.
    """
    if croptype != 0:
        raise ValueError(
            "croptype must be 0: opt_crop=0 pins it there and the crop branches "
            "are not transcribed"
        )
    if dveg not in (1, 3, 4, 7, 8, 9):
        raise ValueError(f"dveg={dveg}: dynamic carbon vegetation is not transcribed")
    if len(laim) != 12 or len(saim) != 12:
        raise ValueError("laim and saim must each have 12 entries")

    snowh = f32(snowh)
    tv = f32(tv)
    lat = f32(lat)
    julian = f32(julian)
    lai = f32(lai)
    sai = f32(sai)
    hvt = f32(hvt)
    hvb = f32(hvb)
    tmin = f32(tmin)

    # --- IF (CROPTYPE == 0) THEN  (always, under opt_crop = 0) -------------
    # IF ( DVEG == 1 .or. DVEG == 3 .or. DVEG == 4 )
    if dveg in (1, 3, 4):
        if lat >= 0.0:
            day = julian
        else:
            # DAY = MOD ( JULIAN + ( 0.5 * YEARLEN ) , REAL(YEARLEN) )
            half = f32(f32(0.5) * f32(yearlen))
            day = f32(math.fmod(f32(julian + half), f32(yearlen)))

        t = f32(f32(f32(12.0) * day) / f32(yearlen))
        it1 = int(f32(t + f32(0.5)))          # REAL -> INTEGER truncates toward 0
        it2 = it1 + 1
        wt1 = f32(f32(float(it1) + f32(0.5)) - t)   # uses the *unclamped* IT1
        wt2 = f32(f32(1.0) - wt1)
        if it1 < 1:
            it1 = 12
        if it2 > 12:
            it2 = 1

        lai = f32(f32(wt1 * f32(laim[it1 - 1])) + f32(wt2 * f32(laim[it2 - 1])))
        sai = f32(f32(wt1 * f32(saim[it1 - 1])) + f32(wt2 * f32(saim[it2 - 1])))

    else:
        sai = _fmax(f32(0.05), f32(f32(0.1) * lai))
        if lai < f32(0.05):
            sai = f32(0.0)

    if sai < f32(0.05):
        sai = 0.0
    if lai < f32(0.05) or sai == 0.0:
        lai = 0.0

    if vegtyp == iswater or vegtyp == isbarren or vegtyp == isice or urban_flag:
        lai = 0.0
        sai = 0.0
    # --- ENDIF  ! CROPTYPE == 0 -------------------------------------------

    # buried by snow
    db = _fmin(_fmax(f32(snowh - hvb), f32(0.0)), f32(hvt - hvb))
    fb = f32(db / _fmax(f32(1.0e-06), f32(hvt - hvb)))

    if hvt > 0.0 and hvt <= 1.0:
        snowhc = f32(hvt * expf(f32(-f32(snowh / f32(0.2)))))
        fb = f32(_fmin(snowh, snowhc) / snowhc)

    elai = f32(lai * f32(f32(1.0) - fb))
    esai = f32(sai * f32(f32(1.0) - fb))
    if esai < f32(0.05):
        esai = 0.0
    if elai < f32(0.05) or esai == 0.0:
        elai = 0.0

    igs = f32(1.0) if tv > tmin else f32(0.0)

    return PhenologyResult(lai=lai, sai=sai, elai=elai, esai=esai, igs=igs, fb=fb)


# ==========================================================================
# PRECIP_HEAT
# ==========================================================================
class PrecipHeatResult(NamedTuple):
    canliq: float   # INTENT(INOUT)
    canice: float   # INTENT(INOUT)
    qintr: float
    qdripr: float
    qthror: float
    qints: float
    qdrips: float
    qthros: float
    pahv: float
    pahg: float
    pahb: float
    qrain: float
    qsnow: float
    snowhin: float
    fwet: float
    cmc: float


def precip_heat(
    *,
    iloc: int,
    jloc: int,
    vegtyp: int,
    ist: int,
    dt: float,
    uu: float,
    vv: float,
    elai: float,
    esai: float,
    fveg: float,
    bdfall: float,
    rain: float,
    snow: float,
    fp: float,
    canliq: float,
    canice: float,
    tv: float,
    sfctmp: float,
    tg: float,
    ch2op: float,
) -> PrecipHeatResult:
    """PRECIP_HEAT, module_sf_noahmplsm.F:1362.

    ``iloc``, ``jloc`` and ``vegtyp`` are accepted because the WRF argument
    list carries them; none of the three appears anywhere in the routine body.
    """
    dt = f32(dt)
    uu = f32(uu)
    vv = f32(vv)
    elai = f32(elai)
    esai = f32(esai)
    fveg = f32(fveg)
    bdfall = f32(bdfall)
    rain = f32(rain)
    snow = f32(snow)
    fp = f32(fp)
    canliq = f32(canliq)
    canice = f32(canice)
    tv = f32(tv)
    sfctmp = f32(sfctmp)
    tg = f32(tg)
    ch2op = f32(ch2op)

    qdripr = f32(0.0)
    qthror = f32(0.0)
    qints = f32(0.0)
    qdrips = f32(0.0)
    qthros = f32(0.0)
    icedrip = f32(0.0)

    lsai = f32(elai + esai)

    # ----------------------- liquid water ---------------------------------
    maxliq = f32(f32(fveg * ch2op) * lsai)

    if lsai > 0.0:
        qintr = f32(f32(fveg * rain) * fp)
        cap = f32(
            f32(f32(maxliq - canliq) / dt)
            * f32(f32(1.0) - expf(f32(-f32(f32(rain * dt) / maxliq))))
        )
        qintr = _fmin(qintr, cap)
        qintr = _fmax(qintr, f32(0.0))
        qdripr = f32(f32(fveg * rain) - qintr)
        qthror = f32(f32(f32(1.0) - fveg) * rain)
        canliq = _fmax(f32(0.0), f32(canliq + f32(qintr * dt)))
    else:
        qintr = f32(0.0)
        qdripr = f32(0.0)
        qthror = rain
        if canliq > 0.0:
            qdripr = f32(qdripr + f32(canliq / dt))
            canliq = f32(0.0)

    # heat transported by liquid water
    pah_ac = f32(f32(f32(fveg * rain) * _CWAT_PER_1000) * f32(sfctmp - tv))
    pah_cg = f32(f32(qdripr * _CWAT_PER_1000) * f32(tv - tg))
    pah_ag = f32(f32(qthror * _CWAT_PER_1000) * f32(sfctmp - tg))

    # ----------------------- canopy ice -----------------------------------
    maxsno = f32(
        f32(f32(fveg * f32(6.6)) * f32(f32(0.27) + f32(f32(46.0) / bdfall))) * lsai
    )

    if lsai > 0.0:
        qints = f32(f32(fveg * snow) * fp)
        cap = f32(
            f32(f32(maxsno - canice) / dt)
            * f32(f32(1.0) - expf(f32(-f32(f32(snow * dt) / maxsno))))
        )
        qints = _fmin(qints, cap)
        qints = _fmax(qints, f32(0.0))
        ft = _fmax(f32(0.0), f32(f32(tv - f32(270.15)) / f32(1.87e5)))
        fv = f32(
            f32(math.sqrt(f32(f32(uu * uu) + f32(vv * vv)))) / f32(1.56e5)
        )
        icedrip = f32(_fmax(f32(0.0), canice) * f32(fv + ft))
        icedrip = _fmin(f32(f32(canice / dt) + qints), icedrip)
        qdrips = f32(f32(f32(fveg * snow) - qints) + icedrip)
        qthros = f32(f32(f32(1.0) - fveg) * snow)
        canice = _fmax(f32(0.0), f32(canice + f32(f32(qints - icedrip) * dt)))
    else:
        qints = f32(0.0)
        qdrips = f32(0.0)
        qthros = snow
        if canice > 0.0:
            qdrips = f32(qdrips + f32(canice / dt))
            canice = f32(0.0)

    # wetted fraction of canopy
    if canice > 0.0:
        fwet = f32(_fmax(f32(0.0), canice) / _fmax(maxsno, f32(1.0e-06)))
    else:
        fwet = f32(_fmax(f32(0.0), canliq) / _fmax(maxliq, f32(1.0e-06)))
    fwet = powf(_fmin(fwet, f32(1.0)), f32(0.667))

    cmc = f32(canliq + canice)

    # heat transported by snow/ice
    pah_ac = f32(
        pah_ac + f32(f32(f32(fveg * snow) * _CICE_PER_1000) * f32(sfctmp - tv))
    )
    pah_cg = f32(pah_cg + f32(f32(qdrips * _CICE_PER_1000) * f32(tv - tg)))
    pah_ag = f32(pah_ag + f32(f32(qthros * _CICE_PER_1000) * f32(sfctmp - tg)))

    pahv = f32(pah_ac - pah_cg)
    pahg = pah_cg
    pahb = pah_ag

    if fveg > 0.0 and fveg < 1.0:
        pahg = f32(pahg / fveg)
        pahb = f32(pahb / f32(f32(1.0) - fveg))
    elif fveg <= 0.0:
        pahb = f32(pahg + pahb)
        pahg = f32(0.0)
        pahv = f32(0.0)
    elif fveg >= 1.0:
        pahb = f32(0.0)

    pahv = _fmax(pahv, f32(-20.0))
    pahv = _fmin(pahv, f32(20.0))
    pahg = _fmax(pahg, f32(-20.0))
    pahg = _fmin(pahg, f32(20.0))
    pahb = _fmax(pahb, f32(-20.0))
    pahb = _fmin(pahb, f32(20.0))

    # rain or snow on the ground
    qrain = f32(qdripr + qthror)
    qsnow = f32(qdrips + qthros)
    snowhin = f32(qsnow / bdfall)

    if ist == 2 and tg > TFRZ:
        qsnow = f32(0.0)
        snowhin = f32(0.0)

    return PrecipHeatResult(
        canliq=canliq,
        canice=canice,
        qintr=qintr,
        qdripr=qdripr,
        qthror=qthror,
        qints=qints,
        qdrips=qdrips,
        qthros=qthros,
        pahv=pahv,
        pahg=pahg,
        pahb=pahb,
        qrain=qrain,
        qsnow=qsnow,
        snowhin=snowhin,
        fwet=fwet,
        cmc=cmc,
    )
