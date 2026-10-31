# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
#   RRTMG longwave/shortwave, transcribed from WRF v4.6.1
#   phys/module_ra_rrtmg_lw.F and phys/module_ra_rrtmg_sw.F, which carry
#   AER's own notice seven and nine times respectively:
#
#       Copyright 2002-2008, Atmospheric & Environmental Research, Inc. (AER).
#       This software may be used, copied, or redistributed as long as it is
#       not sold and this copyright notice is reproduced on each copy made.
#       This model is provided as is without any express or implied warranties.
#                             (http://www.rtweb.aer.com/)
#
#   ArWen takes this material under AER's own current grant instead: BSD
#   3-Clause, "Copyright (c) 2020, Atmospheric and Environmental
#   Research", published by AER at github.com/AER-RC/RRTMG_LW and
#   .../RRTMG_SW.  Text in licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt and
#   beside the packaged coefficients in woof/data/wrf_radiation/.
# ======================================================================
"""Legacy RRTMG shortwave (WRF v4.6.1 ``phys/module_ra_rrtmg_sw.F``, option
``ra_sw_physics = 4``) - NumPy FP32 reference port, held at max_ulp 0 against
the Fortran compiled with gfortran (kind_rb = kind(1.0), i.e. FP32).

Scope and contracts
-------------------
This module ports the SW compute chain routine by routine:

    inatm_sw -> cldprmc_sw -> setcoef_sw -> taumol_sw (14 band routines,
    112 g-points) -> spcvmc_sw -> reftra_sw -> vrtqdr_sw

plus the ``rrtmg_sw`` composition glue (albedo band mapping, cloud/aerosol
transposition, heating rates) and the solar constant / zenith handling of the
WRF option-4 driver.  Every routine is oracle-gated bitwise against fixtures
recorded from the UNMODIFIED Fortran (tools/rrtmg_wrf461_oracle/sw_*).

The McICA sub-column generator is owned by a sibling lane
(``woof.core.rrtmg_mcica.generate_sw_subcolumns``); this module consumes its
outputs (``cldfmcl`` ... ``resnmcl``) as plain inputs.  Coefficient loading
from ``RRTMG_SW_DATA`` is owned by ``woof.ingest.rrtmg_coeffs.
load_rrtmg_sw_coefficients``; :func:`init_sw_tables` consumes that dict (raw
Fortran module variables) and performs the same reduction ``rrtmg_sw_ini``
does (224 -> 112 g-points, exp_tbl, heatfac).  Neither sibling module is
imported at module import time.

Numerics
--------
* Everything is FP32 (numpy float32) with the Fortran's exact expression
  trees; float64 appears only where the Fortran uses doubles (trace-gas
  helper) and inside the glibc transcriptions' internal arithmetic.
* The only runtime libm calls in the whole SW chain are ``log(pavel)`` and
  one ``exp`` in setcoef_sw; both go through the single audited
  transcription of glibc 2.39's ``logf``/``expf`` (the compiler/libc the
  oracle used) in ``woof.core.noahmp_libm``, re-verified here through the
  setcoef fixtures.  ``reftra_sw``/``spcvmc_sw`` use the
  init-built ``exp_tbl`` lookup table, not runtime ``exp``.
* ``earth_sun`` is dead code on the WRF option-4 path (the driver hardcodes
  ``dyofyr = 0``, ``adjes = 1.0`` because WRF's solcon already includes the
  eccentricity adjustment).  :func:`rrtmg_sw` fails closed on ``dyofyr > 0``
  rather than shipping an unverified trig path.

WRF option-4 SW call surface (field-by-field)
---------------------------------------------
Inputs the option-4 driver prepares per column (see RRTMG_SWRAD in the WRF
source, replicated bitwise by tools/rrtmg_wrf461_oracle/sw_fixture_driver):

    play/plev [hPa], tlay/tlev [K] (extra layer to TOA: play(nlay) =
    0.5*plev(nlay), plev(top) = 1e-5 hPa, isothermal extension), tsfc = TSK,
    h2ovmr = max(qv,1e-12)*1.607793 (vmr), o3vmr from o3input=2 climatology
    below model top with shifted-climatology extra layer, co2vmr from
    (280+90*exp(0.02*(yr-2000)))*1e-6 (REAL4 exp!), n2o = 319e-9,
    ch4 = 1774e-9, o2 = 0.209488, asdir=asdif=aldir=aldif = ALBEDO,
    coszen = XCOSZEN (driver skips the call when <= 0), scon =
    SOLCON*(1-OBSCUR), dyofyr=0, adjes=1.0, icld=cldovrlp, iaer=10 with zero
    aerosol arrays for aer_opt=0, inflgsw/iceflgsw/liqflgsw and the McICA
    arrays from the sub-column generator.

Outputs (bottom-to-top, nlay+1 levels): swuflx, swdflx, swuflxc, swdflxc
[W m-2], swhr/swhrc [K day-1] (top layer forced to 0), plus sibvisdir/dif,
sibnirdir/dif, swdkdir/dif/dirc used for SWDDIR/SWVISDIR etc.  The WRF-level
mapping (GSW, SWCF, RTHRATENSW = swhr/86400/pi3d, ...) is provided by
:func:`swrad_option4_outputs`.
"""

from __future__ import annotations

import numpy as np
# numpy >= 2 is essential for the FP32 max_ulp-0 discipline in this
# module: NEP-50 weak promotion keeps float32 op python-scalar in
# float32, while numpy 1.x would silently widen those chains to float64
# and break bitwise parity with the WRF oracle.  Fail closed at import.
if tuple(int(part) for part in np.__version__.split(".")[:2]) < (2, 0):
    raise ImportError(
        f"{__name__} requires numpy >= 2 (NEP-50 weak promotion is part "
        f"of its bitwise FP32 contract); found numpy {np.__version__}")


from woof.core.rrtmg_lw import (
    _libm as _libm,
    logf, expf,
    _r512,
    _lw_dev_chunk as _sw_dev_chunk,
    _lw_dev_mcica as _sw_dev_mcica,
)

F = np.float32

# ---------------------------------------------------------------------------
# parrrsw constants (module parrrsw, WRF v4.6.1)
# ---------------------------------------------------------------------------
NBNDSW = 14
NGPTSW = 112
MXMOL = 38
NMOL = 7
JPB1, JPB2 = 16, 29
RRSW_SCON = F(1.36822e+03)
# All layer arrays use caller-owned workspace; there is no fixed SW ceiling.
MAX_RADIATION_LAYERS = None


def _radiation_layer_count(value):
    from operator import index
    try:
        count = index(value)
    except TypeError as error:
        raise ValueError("radiation layer count must be a positive integer") from error
    if isinstance(value, (bool, np.bool_)) or not 0 < count <= np.iinfo(np.int32).max // NGPTSW:
        raise ValueError("radiation layer count must be positive and fit the CUDA int32 layer/g-point index")
    return count

NSPA = (9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1)      # nspa(16:29)
NSPB = (1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1)      # nspb(16:29)
NGC = (6, 12, 8, 8, 10, 10, 2, 10, 8, 6, 6, 8, 6, 12)  # reduced g per band
NGS = (6, 18, 26, 34, 44, 54, 56, 66, 74, 80, 86, 94, 100, 112)

#: glibc these transcriptions reproduce (see gate tests).
GLIBC_VERSION = "2.39"

# ===========================================================================
# SECTION 1: glibc 2.39 FP32 libm (logf, expf)
#
# The single audited transcription of glibc 2.39's FP32 entry points lives
# in woof.core.noahmp_libm (verified against the live glibc: logf
# 222,414,918 inputs / 0 mismatches; expf 146,800,642 inputs / 2 known
# 1-ULP multiarch residuals at x = 0x4202422F and x = 0xC27C65D9, neither
# producible by setcoef_sw's exp(-1919.4/tavel) (range ~[-10.2, -5.8]) or
# by the exp_tbl build).  The adapters below only fix the carrier type to
# np.float32 for this module's scalar chains; they add no arithmetic.  The
# lane-local generic-C transcriptions that sat here until the integration
# merge were audited bitwise against noahmp_libm over 1.6e6 domain-relevant
# probes (0 mismatches, including both multiarch residual arguments) before
# removal; the fixture decks re-gate the chain end-to-end on every run.
# ===========================================================================


# ===========================================================================
# SECTION 2: coefficient tables container
# ===========================================================================

class SWTables:
    """Post-init RRTMG SW coefficient tables (reduced 112 g-points).

    Attribute names mirror the Fortran module variables.  Band tables live in
    ``kg[16] .. kg[29]`` dicts keyed by the Fortran names (``absa``, ``absb``,
    ``selfref``, ``forref``, ``sfluxref``, ``rayl``, ``strrat``/``strrat1``,
    ``layreffr`` and band extras).  Built either by :func:`tables_from_dump`
    (oracle dump; tests) or :func:`init_sw_tables` (raw coefficient dict from
    the ingest loader; production).
    """

    __slots__ = (
        "kg", "exp_tbl", "bpade", "od_lo", "tblint", "heatfac", "oneminus",
        "grav", "avogad", "pref", "preflog", "tref",
        "extliq1", "ssaliq1", "asyliq1", "extice2", "ssaice2", "asyice2",
        "extice3", "ssaice3", "asyice3", "fdlice3",
        "abari", "bbari", "cbari", "dbari", "ebari", "fbari",
        "ngb", "wavenum1", "wavenum2", "rsrtaua", "rsrpiza", "rsrasya",
        "aer1_climatology", "aer1_optics_sha256",
    )


def tables_from_dump(dump: dict) -> SWTables:
    """Build :class:`SWTables` from the oracle table dump (sw_tables.npz)."""
    t = SWTables()
    t.kg = {}
    for band in range(16, 30):
        p = f"kg{band}/"
        # raw (kao/kbo/...) and reduced (absa/absb/...) names both kept;
        # the compute chain reads reduced ones, init gating reads both.
        t.kg[band] = {k[len(p):]: np.asarray(v)
                      for k, v in dump.items() if k.startswith(p)}
    t.exp_tbl = np.asarray(dump["tbl/exp_tbl"], dtype=np.float32)
    t.bpade = F(dump["tbl/bpade"])
    t.od_lo = F(dump["tbl/od_lo"])
    t.tblint = F(dump["tbl/tblint"])
    t.heatfac = F(dump["con/heatfac"])
    t.oneminus = F(dump["con/oneminus"])
    t.grav = F(dump["con/grav"])
    t.avogad = F(dump["con/avogad"])
    t.pref = np.asarray(dump["ref/pref"], dtype=np.float32)
    t.preflog = np.asarray(dump["ref/preflog"], dtype=np.float32)
    t.tref = np.asarray(dump["ref/tref"], dtype=np.float32)
    for name in ("extliq1", "ssaliq1", "asyliq1", "extice2", "ssaice2",
                 "asyice2", "extice3", "ssaice3", "asyice3", "fdlice3",
                 "abari", "bbari", "cbari", "dbari", "ebari", "fbari"):
        setattr(t, name, np.asarray(dump[f"cld/{name}"], dtype=np.float32))
    t.ngb = np.asarray(dump["wvn/ngb"], dtype=np.int32)
    t.wavenum1 = np.asarray(dump["wvn/wavenum1"], dtype=np.float32)
    t.wavenum2 = np.asarray(dump["wvn/wavenum2"], dtype=np.float32)
    t.rsrtaua = np.asarray(dump["aer/rsrtaua"], dtype=np.float32)
    t.rsrpiza = np.asarray(dump["aer/rsrpiza"], dtype=np.float32)
    t.rsrasya = np.asarray(dump["aer/rsrasya"], dtype=np.float32)
    return t


# ===========================================================================
# SECTION 3: setcoef_sw
# ===========================================================================

def setcoef_sw(tab: SWTables, nlayers, pavel, tavel, coldry, wkl):
    """Port of setcoef_sw.  All array args FP32, Fortran order (1-based in
    the source; 0-based here).  Returns a dict of the Fortran outputs.

    Note: the Fortran computes indbound/tbndfrac/indlev0/t0frac from tbound
    and tz(0) but never uses them in the SW path; they are omitted.
    """
    pavel = np.asarray(pavel, dtype=np.float32)[:nlayers]
    tavel = np.asarray(tavel, dtype=np.float32)[:nlayers]
    coldry = np.asarray(coldry, dtype=np.float32)[:nlayers]
    wkl = np.asarray(wkl, dtype=np.float32)[:, :nlayers]

    preflog, tref = tab.preflog, tab.tref
    stpfac = F(F(296.0) / F(1013.0))

    laytrop = 0
    laylow = 0
    jp = np.zeros(nlayers, dtype=np.int32)
    jt = np.zeros(nlayers, dtype=np.int32)
    jt1 = np.zeros(nlayers, dtype=np.int32)
    colh2o = np.zeros(nlayers, dtype=np.float32)
    colco2 = np.zeros(nlayers, dtype=np.float32)
    colo3 = np.zeros(nlayers, dtype=np.float32)
    coln2o = np.zeros(nlayers, dtype=np.float32)
    colch4 = np.zeros(nlayers, dtype=np.float32)
    colo2 = np.zeros(nlayers, dtype=np.float32)
    colmol = np.zeros(nlayers, dtype=np.float32)
    co2mult = np.zeros(nlayers, dtype=np.float32)
    indself = np.zeros(nlayers, dtype=np.int32)
    indfor = np.zeros(nlayers, dtype=np.int32)
    selffac = np.zeros(nlayers, dtype=np.float32)
    selffrac = np.zeros(nlayers, dtype=np.float32)
    forfac = np.zeros(nlayers, dtype=np.float32)
    forfrac = np.zeros(nlayers, dtype=np.float32)
    fac00 = np.zeros(nlayers, dtype=np.float32)
    fac01 = np.zeros(nlayers, dtype=np.float32)
    fac10 = np.zeros(nlayers, dtype=np.float32)
    fac11 = np.zeros(nlayers, dtype=np.float32)

    for lay in range(nlayers):
        plog = logf(pavel[lay])
        jpv = int(F(F(36.0) - F(5 * F(plog + F(0.04)))))
        # Fortran: jp = int(36. - 5*(plog+0.04)); 5 is integer -> 5.0_rb mult
        if jpv < 1:
            jpv = 1
        elif jpv > 58:
            jpv = 58
        jp[lay] = jpv
        jp1 = jpv + 1
        fp = F(F(5.0) * F(preflog[jpv - 1] - plog))

        jtv = int(F(F(3.0) + F(F(tavel[lay] - tref[jpv - 1]) / F(15.0))))
        if jtv < 1:
            jtv = 1
        elif jtv > 4:
            jtv = 4
        jt[lay] = jtv
        ft = F(F(F(tavel[lay] - tref[jpv - 1]) / F(15.0)) - F(float(jtv - 3)))
        jt1v = int(F(F(3.0) + F(F(tavel[lay] - tref[jp1 - 1]) / F(15.0))))
        if jt1v < 1:
            jt1v = 1
        elif jt1v > 4:
            jt1v = 4
        jt1[lay] = jt1v
        ft1 = F(F(F(tavel[lay] - tref[jp1 - 1]) / F(15.0)) - F(float(jt1v - 3)))

        water = F(wkl[0, lay] / coldry[lay])
        scalefac = F(F(pavel[lay] * stpfac) / tavel[lay])

        if plog <= F(4.56):
            # above laytrop
            forfac[lay] = F(scalefac / F(F(1.0) + water))
            factor = F(F(tavel[lay] - F(188.0)) / F(36.0))
            indfor[lay] = 3
            forfrac[lay] = F(factor - F(1.0))

            colh2o[lay] = F(F(1.0e-20) * wkl[0, lay])
            colco2[lay] = F(F(1.0e-20) * wkl[1, lay])
            colo3[lay] = F(F(1.0e-20) * wkl[2, lay])
            coln2o[lay] = F(F(1.0e-20) * wkl[3, lay])
            colch4[lay] = F(F(1.0e-20) * wkl[5, lay])
            colo2[lay] = F(F(1.0e-20) * wkl[6, lay])
            colmol[lay] = F(F(F(1.0e-20) * coldry[lay]) + colh2o[lay])
            if colco2[lay] == F(0.0):
                colco2[lay] = F(F(1.0e-32) * coldry[lay])
            if coln2o[lay] == F(0.0):
                coln2o[lay] = F(F(1.0e-32) * coldry[lay])
            if colch4[lay] == F(0.0):
                colch4[lay] = F(F(1.0e-32) * coldry[lay])
            if colo2[lay] == F(0.0):
                colo2[lay] = F(F(1.0e-32) * coldry[lay])
            co2reg = F(F(3.55e-24) * coldry[lay])
            co2mult[lay] = F(F(F(F(colco2[lay] - co2reg) * F(272.63)) *
                               expf(F(F(-1919.4) / tavel[lay]))) /
                             F(F(8.7604e-4) * tavel[lay]))
            selffac[lay] = F(0.0)
            selffrac[lay] = F(0.0)
            indself[lay] = 0
        else:
            laytrop += 1
            if plog >= F(6.62):
                laylow += 1
            forfac[lay] = F(scalefac / F(F(1.0) + water))
            factor = F(F(F(332.0) - tavel[lay]) / F(36.0))
            indfor[lay] = min(2, max(1, int(factor)))
            forfrac[lay] = F(factor - F(float(indfor[lay])))

            selffac[lay] = F(water * forfac[lay])
            factor = F(F(tavel[lay] - F(188.0)) / F(7.2))
            indself[lay] = min(9, max(1, int(factor) - 7))
            selffrac[lay] = F(factor - F(float(indself[lay] + 7)))

            colh2o[lay] = F(F(1.0e-20) * wkl[0, lay])
            colco2[lay] = F(F(1.0e-20) * wkl[1, lay])
            colo3[lay] = F(F(1.0e-20) * wkl[2, lay])
            coln2o[lay] = F(F(1.0e-20) * wkl[3, lay])
            colch4[lay] = F(F(1.0e-20) * wkl[5, lay])
            colo2[lay] = F(F(1.0e-20) * wkl[6, lay])
            colmol[lay] = F(F(F(1.0e-20) * coldry[lay]) + colh2o[lay])
            if colco2[lay] == F(0.0):
                colco2[lay] = F(F(1.0e-32) * coldry[lay])
            if coln2o[lay] == F(0.0):
                coln2o[lay] = F(F(1.0e-32) * coldry[lay])
            if colch4[lay] == F(0.0):
                colch4[lay] = F(F(1.0e-32) * coldry[lay])
            if colo2[lay] == F(0.0):
                colo2[lay] = F(F(1.0e-32) * coldry[lay])
            co2reg = F(F(3.55e-24) * coldry[lay])
            co2mult[lay] = F(F(F(F(colco2[lay] - co2reg) * F(272.63)) *
                               expf(F(F(-1919.4) / tavel[lay]))) /
                             F(F(8.7604e-4) * tavel[lay]))

        compfp = F(F(1.0) - fp)
        fac10[lay] = F(compfp * ft)
        fac00[lay] = F(compfp * F(F(1.0) - ft))
        fac11[lay] = F(fp * ft1)
        fac01[lay] = F(fp * F(F(1.0) - ft1))

    return dict(laytrop=laytrop, layswtch=0, laylow=laylow, jp=jp, jt=jt,
                jt1=jt1, colh2o=colh2o, colco2=colco2, colo3=colo3,
                coln2o=coln2o, colch4=colch4, colo2=colo2, colmol=colmol,
                co2mult=co2mult, selffac=selffac, selffrac=selffrac,
                indself=indself, indfor=indfor, forfac=forfac,
                forfrac=forfrac, fac00=fac00, fac01=fac01, fac10=fac10,
                fac11=fac11)


# ===========================================================================
# SECTION 4: taumol_sw (14 band routines, 112 g-points)
# ===========================================================================

def _spec_facs(colA, colB, strrat, oneminus, mult, fac00, fac01, fac10, fac11):
    """Common speccomb/specparm/js/fs/fac### block (scalar, one layer)."""
    speccomb = F(colA + F(strrat * colB))
    specparm = F(colA / speccomb)
    if specparm >= oneminus:
        specparm = oneminus
    specmult = F(F(mult) * specparm)
    js = 1 + int(specmult)
    fs = F(specmult - np.trunc(specmult))     # mod(specmult, 1._rb)
    one = F(1.0)
    return (speccomb, js, fs,
            F(F(one - fs) * fac00), F(F(one - fs) * fac10),
            F(fs * fac00), F(fs * fac10),
            F(F(one - fs) * fac01), F(F(one - fs) * fac11),
            F(fs * fac01), F(fs * fac11))


def _acc8(a, ind0, ind1, dind, f000, f100, f010, f110, f001, f101, f011, f111):
    """Fortran left-to-right sum of the 8 interpolation terms, vector over g.

    The Fortran expression is a single sum evaluated left to right:
    f000*a(ind0,:) + f100*a(ind0+1,:) + f010*a(ind0+d,:) + f110*a(ind0+d+1,:)
    + f001*a(ind1,:) + f101*a(ind1+1,:) + f011*a(ind1+d,:) + f111*a(ind1+d+1,:)
    """
    s = F(f000) * a[ind0 - 1]
    s = F(s + F(F(f100) * a[ind0]))
    s = F(s + F(F(f010) * a[ind0 - 1 + dind]))
    s = F(s + F(F(f110) * a[ind0 + dind]))
    s = F(s + F(F(f001) * a[ind1 - 1]))
    s = F(s + F(F(f101) * a[ind1]))
    s = F(s + F(F(f011) * a[ind1 - 1 + dind]))
    s = F(s + F(F(f111) * a[ind1 + dind]))
    return s


def _acc4(a, ind0, ind1, f00, f10, f01, f11):
    """fac00*a(ind0,:) + fac10*a(ind0+1,:) + fac01*a(ind1,:) + fac11*a(ind1+1,:)"""
    s = F(f00) * a[ind0 - 1]
    s = F(s + F(F(f10) * a[ind0]))
    s = F(s + F(F(f01) * a[ind1 - 1]))
    s = F(s + F(F(f11) * a[ind1]))
    return s


def _selffor(colh2o, selffac, selffrac, inds, forfac, forfrac, indf,
             selfref, forref):
    """colh2o * (selffac*(selfref(inds)+selffrac*(selfref(inds+1)-selfref(inds)))
    + forfac*(forref(indf)+forfrac*(forref(indf+1)-forref(indf))))"""
    sr = F(selfref[inds - 1] + F(F(selffrac) *
                                 F(selfref[inds] - selfref[inds - 1])))
    fr = F(forref[indf - 1] + F(F(forfrac) *
                                F(forref[indf] - forref[indf - 1])))
    return F(F(colh2o) * F(F(F(selffac) * sr) + F(F(forfac) * fr)))


def _forr(colh2o, forfac, forfrac, indf, forref):
    # Fortran: colh2o(lay) * forfac(lay) * (forref(indf)+forfrac*(...)),
    # which associates as ((colh2o*forfac) * fr).
    fr = F(forref[indf - 1] + F(F(forfrac) *
                                F(forref[indf] - forref[indf - 1])))
    return F(F(F(colh2o) * F(forfac)) * fr)


def taumol_sw(tab: SWTables, nlayers, colh2o, colco2, colch4, colo2, colo3,
              colmol, laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor):
    """Port of taumol_sw.  Returns (sfluxzen[112], taug[nlay,112],
    taur[nlay,112]) as FP32 arrays.  All index arrays carry Fortran values."""
    oneminus = tab.oneminus
    sfluxzen = np.zeros(NGPTSW, dtype=np.float32)
    taug = np.zeros((nlayers, NGPTSW), dtype=np.float32)
    taur = np.zeros((nlayers, NGPTSW), dtype=np.float32)

    args = (tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
            laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
            selffac, selffrac, indself, forfac, forfrac, indfor,
            sfluxzen, taug, taur, oneminus)
    _taumol16(*args)
    _taumol17(*args)
    _taumol18(*args)
    _taumol19(*args)
    _taumol20(*args)
    _taumol21(*args)
    _taumol22(*args)
    _taumol23(*args)
    _taumol24(*args)
    _taumol25(*args)
    _taumol26(*args)
    _taumol27(*args)
    _taumol28(*args)
    _taumol29(*args)
    return sfluxzen, taug, taur


def _band(tab, n):
    return tab.kg[n]


def _laysolfr_lower(laytrop, jp, layreffr):
    laysolfr = laytrop
    for lay in range(1, laytrop + 1):
        if jp[lay - 1] < layreffr and jp[lay] >= layreffr:
            laysolfr = min(lay + 1, laytrop)
    return laysolfr


def _laysolfr_upper(laytrop, nlayers, jp, layreffr):
    laysolfr = nlayers
    for lay in range(laytrop + 1, nlayers + 1):
        if jp[lay - 2] < layreffr and jp[lay - 1] >= layreffr:
            laysolfr = lay
    return laysolfr


def _taumol16(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 16)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    rayl, strrat1, layreffr = F(kg["rayl"]), F(kg["strrat1"]), int(kg["layreffr"])
    g0 = 0
    for lay in range(1, laytrop + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colch4[L], strrat1, oneminus, 8.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[0] + js
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[0] + js
        inds, indf = indself[L], indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc8(absa, ind0, ind1, 9, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        sf = _selffor(colh2o[L], selffac[L], selffrac[L], inds,
                      forfac[L], forfrac[L], indf, selfref, forref)
        taug[L, g0:g0 + 6] = F(F(F(speccomb) * core) + sf)
        taur[L, g0:g0 + 6] = tauray
    laysolfr = _laysolfr_upper(laytrop, nlayers, jp, layreffr)
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[0] + 1
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[0] + 1
        tauray = F(colmol[L] * rayl)
        core = _acc4(absb, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 6] = F(F(colch4[L]) * core)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 6] = sflux
        taur[L, g0:g0 + 6] = tauray


def _taumol17(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 17)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    rayl, strrat, layreffr = F(kg["rayl"]), F(kg["strrat"]), int(kg["layreffr"])
    g0 = NGS[0]
    for lay in range(1, laytrop + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colco2[L], strrat, oneminus, 8.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[1] + js
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[1] + js
        inds, indf = indself[L], indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc8(absa, ind0, ind1, 9, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        sf = _selffor(colh2o[L], selffac[L], selffrac[L], inds,
                      forfac[L], forfrac[L], indf, selfref, forref)
        taug[L, g0:g0 + 12] = F(F(F(speccomb) * core) + sf)
        taur[L, g0:g0 + 12] = tauray
    laysolfr = _laysolfr_upper(laytrop, nlayers, jp, layreffr)
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colco2[L], strrat, oneminus, 4.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[1] + js
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[1] + js
        indf = indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc8(absb, ind0, ind1, 5, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        fr = _forr(colh2o[L], forfac[L], forfrac[L], indf, forref)
        taug[L, g0:g0 + 12] = F(F(F(speccomb) * core) + fr)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 12] = F(sflux[:, js - 1] +
                                     F(F(fs) * F(sflux[:, js] - sflux[:, js - 1])))
        taur[L, g0:g0 + 12] = tauray


def _taumol18(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 18)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    rayl, strrat, layreffr = F(kg["rayl"]), F(kg["strrat"]), int(kg["layreffr"])
    g0 = NGS[1]
    laysolfr = _laysolfr_lower(laytrop, jp, layreffr)
    for lay in range(1, laytrop + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colch4[L], strrat, oneminus, 8.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[2] + js
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[2] + js
        inds, indf = indself[L], indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc8(absa, ind0, ind1, 9, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        sf = _selffor(colh2o[L], selffac[L], selffrac[L], inds,
                      forfac[L], forfrac[L], indf, selfref, forref)
        taug[L, g0:g0 + 8] = F(F(F(speccomb) * core) + sf)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 8] = F(sflux[:, js - 1] +
                                    F(F(fs) * F(sflux[:, js] - sflux[:, js - 1])))
        taur[L, g0:g0 + 8] = tauray
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[2] + 1
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[2] + 1
        tauray = F(colmol[L] * rayl)
        core = _acc4(absb, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 8] = F(F(colch4[L]) * core)
        taur[L, g0:g0 + 8] = tauray


def _taumol19(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 19)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    rayl, strrat, layreffr = F(kg["rayl"]), F(kg["strrat"]), int(kg["layreffr"])
    g0 = NGS[2]
    laysolfr = _laysolfr_lower(laytrop, jp, layreffr)
    for lay in range(1, laytrop + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colco2[L], strrat, oneminus, 8.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[3] + js
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[3] + js
        inds, indf = indself[L], indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc8(absa, ind0, ind1, 9, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        sf = _selffor(colh2o[L], selffac[L], selffrac[L], inds,
                      forfac[L], forfrac[L], indf, selfref, forref)
        taug[L, g0:g0 + 8] = F(F(F(speccomb) * core) + sf)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 8] = F(sflux[:, js - 1] +
                                    F(F(fs) * F(sflux[:, js] - sflux[:, js - 1])))
        taur[L, g0:g0 + 8] = tauray
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[3] + 1
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[3] + 1
        tauray = F(colmol[L] * rayl)
        core = _acc4(absb, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 8] = F(F(colco2[L]) * core)
        taur[L, g0:g0 + 8] = tauray


def _taumol20(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 20)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    absch4 = kg["absch4"]
    rayl, layreffr = F(kg["rayl"]), int(kg["layreffr"])
    g0 = NGS[3]
    laysolfr = _laysolfr_lower(laytrop, jp, layreffr)
    for lay in range(1, laytrop + 1):
        L = lay - 1
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[4] + 1
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[4] + 1
        inds, indf = indself[L], indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc4(absa, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        sr = F(selfref[inds - 1] + F(F(selffrac[L]) *
                                     F(selfref[inds] - selfref[inds - 1])))
        fr = F(forref[indf - 1] + F(F(forfrac[L]) *
                                    F(forref[indf] - forref[indf - 1])))
        inner = F(F(core + F(F(selffac[L]) * sr)) + F(F(forfac[L]) * fr))
        taug[L, g0:g0 + 10] = F(F(F(colh2o[L]) * inner) +
                                F(F(colch4[L]) * absch4))
        taur[L, g0:g0 + 10] = tauray
        if lay == laysolfr:
            sfluxzen[g0:g0 + 10] = sflux
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[4] + 1
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[4] + 1
        indf = indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc4(absb, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        fr = F(forref[indf - 1] + F(F(forfrac[L]) *
                                    F(forref[indf] - forref[indf - 1])))
        inner = F(core + F(F(forfac[L]) * fr))
        taug[L, g0:g0 + 10] = F(F(F(colh2o[L]) * inner) +
                                F(F(colch4[L]) * absch4))
        taur[L, g0:g0 + 10] = tauray


def _taumol21(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 21)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    rayl, strrat, layreffr = F(kg["rayl"]), F(kg["strrat"]), int(kg["layreffr"])
    g0 = NGS[4]
    laysolfr = _laysolfr_lower(laytrop, jp, layreffr)
    for lay in range(1, laytrop + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colco2[L], strrat, oneminus, 8.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[5] + js
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[5] + js
        inds, indf = indself[L], indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc8(absa, ind0, ind1, 9, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        sf = _selffor(colh2o[L], selffac[L], selffrac[L], inds,
                      forfac[L], forfrac[L], indf, selfref, forref)
        taug[L, g0:g0 + 10] = F(F(F(speccomb) * core) + sf)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 10] = F(sflux[:, js - 1] +
                                     F(F(fs) * F(sflux[:, js] - sflux[:, js - 1])))
        taur[L, g0:g0 + 10] = tauray
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colco2[L], strrat, oneminus, 4.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[5] + js
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[5] + js
        indf = indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc8(absb, ind0, ind1, 5, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        fr = _forr(colh2o[L], forfac[L], forfrac[L], indf, forref)
        taug[L, g0:g0 + 10] = F(F(F(speccomb) * core) + fr)
        taur[L, g0:g0 + 10] = tauray


def _taumol22(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 22)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    rayl, strrat, layreffr = F(kg["rayl"]), F(kg["strrat"]), int(kg["layreffr"])
    g0 = NGS[5]
    o2adj = F(1.6)
    laysolfr = _laysolfr_lower(laytrop, jp, layreffr)
    for lay in range(1, laytrop + 1):
        L = lay - 1
        o2cont = F(F(F(4.35e-4) * colo2[L]) / F(F(350.0) * F(2.0)))
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colo2[L], F(o2adj * strrat), oneminus, 8.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[6] + js
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[6] + js
        inds, indf = indself[L], indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc8(absa, ind0, ind1, 9, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        sf = _selffor(colh2o[L], selffac[L], selffrac[L], inds,
                      forfac[L], forfrac[L], indf, selfref, forref)
        taug[L, g0:g0 + 2] = F(F(F(F(speccomb) * core) + sf) + o2cont)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 2] = F(sflux[:, js - 1] +
                                    F(F(fs) * F(sflux[:, js] - sflux[:, js - 1])))
        taur[L, g0:g0 + 2] = tauray
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        o2cont = F(F(F(4.35e-4) * colo2[L]) / F(F(350.0) * F(2.0)))
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[6] + 1
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[6] + 1
        tauray = F(colmol[L] * rayl)
        core = _acc4(absb, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 2] = F(F(F(F(colo2[L]) * o2adj) * core) + o2cont)
        taur[L, g0:g0 + 2] = tauray


def _taumol23(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 23)
    absa = kg["absa"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    rayl, givfac, layreffr = kg["rayl"], F(kg["givfac"]), int(kg["layreffr"])
    g0 = NGS[6]
    laysolfr = _laysolfr_lower(laytrop, jp, layreffr)
    for lay in range(1, laytrop + 1):
        L = lay - 1
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[7] + 1
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[7] + 1
        inds, indf = indself[L], indfor[L]
        tauray = F(F(colmol[L]) * rayl)
        core = _acc4(absa, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        sr = F(selfref[inds - 1] + F(F(selffrac[L]) *
                                     F(selfref[inds] - selfref[inds - 1])))
        fr = F(forref[indf - 1] + F(F(forfrac[L]) *
                                    F(forref[indf] - forref[indf - 1])))
        inner = F(F(F(givfac * core) + F(F(selffac[L]) * sr)) +
                  F(F(forfac[L]) * fr))
        taug[L, g0:g0 + 10] = F(F(colh2o[L]) * inner)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 10] = sflux
        taur[L, g0:g0 + 10] = tauray
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        taug[L, g0:g0 + 10] = F(0.0)
        taur[L, g0:g0 + 10] = F(F(colmol[L]) * rayl)


def _taumol24(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 24)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    abso3a, abso3b = kg["abso3a"], kg["abso3b"]
    rayla, raylb = kg["rayla"], kg["raylb"]
    strrat, layreffr = F(kg["strrat"]), int(kg["layreffr"])
    g0 = NGS[7]
    laysolfr = _laysolfr_lower(laytrop, jp, layreffr)
    for lay in range(1, laytrop + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colh2o[L], colo2[L], strrat, oneminus, 8.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[8] + js
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[8] + js
        inds, indf = indself[L], indfor[L]
        tauray = F(F(colmol[L]) *
                   F(rayla[:, js - 1] +
                     F(F(fs) * F(rayla[:, js] - rayla[:, js - 1]))))
        core = _acc8(absa, ind0, ind1, 9, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        sf = _selffor(colh2o[L], selffac[L], selffrac[L], inds,
                      forfac[L], forfrac[L], indf, selfref, forref)
        taug[L, g0:g0 + 8] = F(F(F(F(F(speccomb) * core) +
                                   F(F(colo3[L]) * abso3a)) + sf))
        if lay == laysolfr:
            sfluxzen[g0:g0 + 8] = F(sflux[:, js - 1] +
                                    F(F(fs) * F(sflux[:, js] - sflux[:, js - 1])))
        taur[L, g0:g0 + 8] = tauray
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[8] + 1
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[8] + 1
        tauray = F(F(colmol[L]) * raylb)
        core = _acc4(absb, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 8] = F(F(F(colo2[L]) * core) +
                               F(F(colo3[L]) * abso3b))
        taur[L, g0:g0 + 8] = tauray


def _taumol25(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 25)
    absa = kg["absa"]
    sflux = kg["sfluxref"]
    abso3a, abso3b = kg["abso3a"], kg["abso3b"]
    rayl, layreffr = kg["rayl"], int(kg["layreffr"])
    g0 = NGS[8]
    laysolfr = _laysolfr_lower(laytrop, jp, layreffr)
    for lay in range(1, laytrop + 1):
        L = lay - 1
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[9] + 1
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[9] + 1
        tauray = F(F(colmol[L]) * rayl)
        core = _acc4(absa, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 6] = F(F(F(colh2o[L]) * core) +
                               F(F(colo3[L]) * abso3a))
        if lay == laysolfr:
            sfluxzen[g0:g0 + 6] = sflux
        taur[L, g0:g0 + 6] = tauray
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        taug[L, g0:g0 + 6] = F(F(colo3[L]) * abso3b)
        taur[L, g0:g0 + 6] = F(F(colmol[L]) * rayl)


def _taumol26(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 26)
    sflux, rayl = kg["sfluxref"], kg["rayl"]
    g0 = NGS[9]
    laysolfr = laytrop
    for lay in range(1, laytrop + 1):
        L = lay - 1
        if lay == laysolfr:
            sfluxzen[g0:g0 + 6] = sflux
        taug[L, g0:g0 + 6] = F(0.0)
        taur[L, g0:g0 + 6] = F(F(colmol[L]) * rayl)
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        taug[L, g0:g0 + 6] = F(0.0)
        taur[L, g0:g0 + 6] = F(F(colmol[L]) * rayl)


def _taumol27(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 27)
    absa, absb = kg["absa"], kg["absb"]
    sflux, rayl = kg["sfluxref"], kg["rayl"]
    scalekur, layreffr = F(kg["scalekur"]), int(kg["layreffr"])
    g0 = NGS[10]
    for lay in range(1, laytrop + 1):
        L = lay - 1
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[11] + 1
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[11] + 1
        tauray = F(F(colmol[L]) * rayl)
        core = _acc4(absa, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 8] = F(F(colo3[L]) * core)
        taur[L, g0:g0 + 8] = tauray
    laysolfr = _laysolfr_upper(laytrop, nlayers, jp, layreffr)
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[11] + 1
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[11] + 1
        tauray = F(F(colmol[L]) * rayl)
        core = _acc4(absb, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 8] = F(F(colo3[L]) * core)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 8] = F(scalekur * sflux)
        taur[L, g0:g0 + 8] = tauray


def _taumol28(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 28)
    absa, absb = kg["absa"], kg["absb"]
    sflux = kg["sfluxref"]
    rayl, strrat, layreffr = F(kg["rayl"]), F(kg["strrat"]), int(kg["layreffr"])
    g0 = NGS[11]
    for lay in range(1, laytrop + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colo3[L], colo2[L], strrat, oneminus, 8.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[12] + js
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[12] + js
        tauray = F(colmol[L] * rayl)
        core = _acc8(absa, ind0, ind1, 9, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        taug[L, g0:g0 + 6] = F(F(speccomb) * core)
        taur[L, g0:g0 + 6] = tauray
    laysolfr = _laysolfr_upper(laytrop, nlayers, jp, layreffr)
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        (speccomb, js, fs, f000, f010, f100, f110,
         f001, f011, f101, f111) = _spec_facs(
            colo3[L], colo2[L], strrat, oneminus, 4.0,
            fac00[L], fac01[L], fac10[L], fac11[L])
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[12] + js
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[12] + js
        tauray = F(colmol[L] * rayl)
        core = _acc8(absb, ind0, ind1, 5, f000, f100, f010, f110,
                     f001, f101, f011, f111)
        taug[L, g0:g0 + 6] = F(F(speccomb) * core)
        if lay == laysolfr:
            sfluxzen[g0:g0 + 6] = F(sflux[:, js - 1] +
                                    F(F(fs) * F(sflux[:, js] - sflux[:, js - 1])))
        taur[L, g0:g0 + 6] = tauray


def _taumol29(tab, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol,
              laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor,
              sfluxzen, taug, taur, oneminus):
    kg = _band(tab, 29)
    absa, absb = kg["absa"], kg["absb"]
    selfref, forref, sflux = kg["selfref"], kg["forref"], kg["sfluxref"]
    absh2o, absco2 = kg["absh2o"], kg["absco2"]
    rayl, layreffr = F(kg["rayl"]), int(kg["layreffr"])
    g0 = NGS[12]
    for lay in range(1, laytrop + 1):
        L = lay - 1
        ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * NSPA[13] + 1
        ind1 = (jp[L] * 5 + (jt1[L] - 1)) * NSPA[13] + 1
        inds, indf = indself[L], indfor[L]
        tauray = F(colmol[L] * rayl)
        core = _acc4(absa, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        sr = F(selfref[inds - 1] + F(F(selffrac[L]) *
                                     F(selfref[inds] - selfref[inds - 1])))
        fr = F(forref[indf - 1] + F(F(forfrac[L]) *
                                    F(forref[indf] - forref[indf - 1])))
        inner = F(F(core + F(F(selffac[L]) * sr)) + F(F(forfac[L]) * fr))
        taug[L, g0:g0 + 12] = F(F(F(colh2o[L]) * inner) +
                                F(F(colco2[L]) * absco2))
        taur[L, g0:g0 + 12] = tauray
    laysolfr = _laysolfr_upper(laytrop, nlayers, jp, layreffr)
    for lay in range(laytrop + 1, nlayers + 1):
        L = lay - 1
        ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * NSPB[13] + 1
        ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * NSPB[13] + 1
        tauray = F(colmol[L] * rayl)
        core = _acc4(absb, ind0, ind1, fac00[L], fac10[L], fac01[L], fac11[L])
        taug[L, g0:g0 + 12] = F(F(F(colco2[L]) * core) +
                                F(F(colh2o[L]) * absh2o))
        if lay == laysolfr:
            sfluxzen[g0:g0 + 12] = sflux
        taur[L, g0:g0 + 12] = tauray


# ===========================================================================
# SECTION 5: cldprmc_sw (cloud optics for McICA)
# ===========================================================================

def cldprmc_sw(tab: SWTables, nlayers, inflag, iceflag, liqflag, cldfmc,
               ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc,
               taucmc, ssacmc, asmcmc, fsfcmc):
    """Port of cldprmc_sw.  Arrays are (ngptsw, nlayers) FP32; taucmc,
    ssacmc, asmcmc are updated in place (inout in the Fortran); taormc is
    returned.  Fortran STOP/wrf_error_fatal conditions raise ValueError."""
    cldmin = F(1.0e-20)
    eps = F(1.0e-06)
    one = F(1.0)
    taormc = taucmc.copy()

    ngb = tab.ngb
    extice2, ssaice2, asyice2 = tab.extice2, tab.ssaice2, tab.asyice2
    extice3, ssaice3, asyice3 = tab.extice3, tab.ssaice3, tab.asyice3
    fdlice3 = tab.fdlice3
    extliq1, ssaliq1, asyliq1 = tab.extliq1, tab.ssaliq1, tab.asyliq1
    abari, bbari = tab.abari, tab.bbari
    cbari, dbari, ebari, fbari = tab.cbari, tab.dbari, tab.ebari, tab.fbari
    wavenum2 = tab.wavenum2

    if inflag == 1:
        raise ValueError("INFLAG = 1 OPTION NOT AVAILABLE WITH MCICA")

    for lay in range(nlayers):
        for ig in range(NGPTSW):
            cwp = F(F(ciwpmc[ig, lay] + clwpmc[ig, lay]) + cswpmc[ig, lay])
            if not (cldfmc[ig, lay] >= cldmin and
                    (cwp >= cldmin or taucmc[ig, lay] >= cldmin)):
                continue

            if inflag == 0:
                taucldorig_a = taucmc[ig, lay]
                ffp = fsfcmc[ig, lay]
                ffp1 = F(one - ffp)
                ffpssa = F(one - F(ffp * ssacmc[ig, lay]))
                ssacloud_a = F(F(ffp1 * ssacmc[ig, lay]) / ffpssa)
                taucloud_a = F(ffpssa * taucldorig_a)
                taormc[ig, lay] = taucldorig_a
                ssacmc[ig, lay] = ssacloud_a
                taucmc[ig, lay] = taucloud_a
                asmcmc[ig, lay] = F(F(asmcmc[ig, lay] - ffp) / ffp1)
                continue

            # inflag >= 2
            radice = reicmc[lay]
            if F(ciwpmc[ig, lay] + cswpmc[ig, lay]) == F(0.0):
                extcoice = F(0.0); ssacoice = F(0.0)
                gice = F(0.0); forwice = F(0.0)
                extcosno = F(0.0); ssacosno = F(0.0)
                gsno = F(0.0); forwsno = F(0.0)
            elif iceflag == 1:
                ib = int(ngb[ig])
                w2 = wavenum2[ib - 16]
                if w2 > F(1.43e04):
                    icx = 1
                elif w2 > F(7.7e03):
                    icx = 2
                elif w2 > F(5.3e03):
                    icx = 3
                elif w2 > F(4.0e03):
                    icx = 4
                elif w2 >= F(2.5e03):
                    icx = 5
                else:
                    raise ValueError("iceflag=1: wavenum2 out of range")
                extcoice = F(abari[icx - 1] + F(bbari[icx - 1] / radice))
                ssacoice = F(F(one - cbari[icx - 1]) -
                             F(dbari[icx - 1] * radice))
                gice = F(ebari[icx - 1] + F(fbari[icx - 1] * radice))
                if gice >= one:
                    gice = F(one - eps)
                forwice = F(gice * gice)
                if extcoice < F(0.0):
                    raise ValueError("ICE EXTINCTION LESS THAN 0.0")
                if ssacoice > one or ssacoice < F(0.0):
                    raise ValueError("ICE SSA OUT OF RANGE")
                if gice > one or gice < F(0.0):
                    raise ValueError("ICE ASYM OUT OF RANGE")
                extcosno = F(0.0); ssacosno = F(0.0)
                gsno = F(0.0); forwsno = F(0.0)
            elif iceflag == 2:
                if radice < F(5.0) or radice > F(131.0):
                    raise ValueError("ICE RADIUS OUT OF BOUNDS")
                factor = F(F(radice - F(2.0)) / F(3.0))
                index = int(factor)
                if index == 43:
                    index = 42
                fint = F(factor - F(float(index)))
                ib = int(ngb[ig])
                c = ib - 16
                extcoice = F(extice2[index - 1, c] + F(fint *
                             F(extice2[index, c] - extice2[index - 1, c])))
                ssacoice = F(ssaice2[index - 1, c] + F(fint *
                             F(ssaice2[index, c] - ssaice2[index - 1, c])))
                gice = F(asyice2[index - 1, c] + F(fint *
                         F(asyice2[index, c] - asyice2[index - 1, c])))
                forwice = F(gice * gice)
                if extcoice < F(0.0):
                    raise ValueError("ICE EXTINCTION LESS THAN 0.0")
                if ssacoice > one or ssacoice < F(0.0):
                    raise ValueError("ICE SSA OUT OF RANGE")
                if gice > one or gice < F(0.0):
                    raise ValueError("ICE ASYM OUT OF RANGE")
                extcosno = F(0.0); ssacosno = F(0.0)
                gsno = F(0.0); forwsno = F(0.0)
            else:  # iceflag >= 3
                if radice < F(5.0) or radice > F(140.0):
                    raise ValueError(
                        "ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS")
                factor = F(F(radice - F(2.0)) / F(3.0))
                index = int(factor)
                if index == 46:
                    index = 45
                fint = F(factor - F(float(index)))
                ib = int(ngb[ig])
                c = ib - 16
                extcoice = F(extice3[index - 1, c] + F(fint *
                             F(extice3[index, c] - extice3[index - 1, c])))
                ssacoice = F(ssaice3[index - 1, c] + F(fint *
                             F(ssaice3[index, c] - ssaice3[index - 1, c])))
                gice = F(asyice3[index - 1, c] + F(fint *
                         F(asyice3[index, c] - asyice3[index - 1, c])))
                fdelta = F(fdlice3[index - 1, c] + F(fint *
                           F(fdlice3[index, c] - fdlice3[index - 1, c])))
                if fdelta < F(0.0):
                    raise ValueError("FDELTA LESS THAN 0.0")
                if fdelta > one:
                    raise ValueError("FDELTA GT THAN 1.0")
                forwice = F(fdelta + F(F(0.5) / ssacoice))
                if forwice > gice:
                    forwice = gice
                if extcoice < F(0.0):
                    raise ValueError("ICE EXTINCTION LESS THAN 0.0")
                if ssacoice > one or ssacoice < F(0.0):
                    raise ValueError("ICE SSA OUT OF RANGE")
                if gice > one or gice < F(0.0):
                    raise ValueError("ICE ASYM OUT OF RANGE")
                extcosno = F(0.0); ssacosno = F(0.0)
                gsno = F(0.0); forwsno = F(0.0)

            # snow (Fortran: separate IF after the ice block)
            if cswpmc[ig, lay] > F(0.0) and iceflag == 5:
                radsno = resnmc[lay]
                if radsno < F(5.0) or radsno > F(140.0):
                    raise ValueError(
                        "SNOW GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS")
                factor = F(F(radsno - F(2.0)) / F(3.0))
                index = int(factor)
                if index == 46:
                    index = 45
                fint = F(factor - F(float(index)))
                ib = int(ngb[ig])
                c = ib - 16
                extcosno = F(extice3[index - 1, c] + F(fint *
                             F(extice3[index, c] - extice3[index - 1, c])))
                ssacosno = F(ssaice3[index - 1, c] + F(fint *
                             F(ssaice3[index, c] - ssaice3[index - 1, c])))
                gsno = F(asyice3[index - 1, c] + F(fint *
                         F(asyice3[index, c] - asyice3[index - 1, c])))
                fdelta = F(fdlice3[index - 1, c] + F(fint *
                           F(fdlice3[index, c] - fdlice3[index - 1, c])))
                if fdelta < F(0.0):
                    raise ValueError("FDELTA LESS THAN 0.0")
                if fdelta > one:
                    raise ValueError("FDELTA GT THAN 1.0")
                forwsno = F(fdelta + F(F(0.5) / ssacosno))
                if forwsno > gsno:
                    forwsno = gsno
                if extcosno < F(0.0):
                    raise ValueError("SNOW EXTINCTION LESS THAN 0.0")
                if ssacosno > one or ssacosno < F(0.0):
                    raise ValueError("SNOW SSA OUT OF RANGE")
                if gsno > one or gsno < F(0.0):
                    raise ValueError("SNOW ASYM OUT OF RANGE")
            else:
                extcosno = F(0.0); ssacosno = F(0.0)
                gsno = F(0.0); forwsno = F(0.0)

            # liquid
            if clwpmc[ig, lay] == F(0.0):
                extcoliq = F(0.0); ssacoliq = F(0.0)
                gliq = F(0.0); forwliq = F(0.0)
            elif liqflag == 1:
                radliq = relqmc[lay]
                if radliq < F(1.5) or radliq > F(60.0):
                    raise ValueError("liquid effective radius out of bounds")
                index = int(F(radliq - F(1.5)))
                if index == 0:
                    index = 1
                if index == 58:
                    index = 57
                fint = F(F(radliq - F(1.5)) - F(float(index)))
                ib = int(ngb[ig])
                c = ib - 16
                extcoliq = F(extliq1[index - 1, c] + F(fint *
                             F(extliq1[index, c] - extliq1[index - 1, c])))
                ssacoliq = F(ssaliq1[index - 1, c] + F(fint *
                             F(ssaliq1[index, c] - ssaliq1[index - 1, c])))
                if fint < F(0.0) and ssacoliq > one:
                    ssacoliq = ssaliq1[index - 1, c]
                gliq = F(asyliq1[index - 1, c] + F(fint *
                         F(asyliq1[index, c] - asyliq1[index - 1, c])))
                forwliq = F(gliq * gliq)
                if extcoliq < F(0.0):
                    raise ValueError("LIQUID EXTINCTION LESS THAN 0.0")
                if ssacoliq > one or ssacoliq < F(0.0):
                    raise ValueError("LIQUID SSA OUT OF RANGE")
                if gliq > one or gliq < F(0.0):
                    raise ValueError("LIQUID ASYM OUT OF RANGE")
            else:
                raise ValueError("liqflag != 1 not reachable via option 4")

            if iceflag < 5:
                tauliqorig = F(clwpmc[ig, lay] * extcoliq)
                tauiceorig = F(ciwpmc[ig, lay] * extcoice)
                taormc[ig, lay] = F(tauliqorig + tauiceorig)
                ssaliq = F(F(ssacoliq * F(one - forwliq)) /
                           F(one - F(forwliq * ssacoliq)))
                tauliq = F(F(one - F(forwliq * ssacoliq)) * tauliqorig)
                ssaice = F(F(ssacoice * F(one - forwice)) /
                           F(one - F(forwice * ssacoice)))
                tauice = F(F(one - F(forwice * ssacoice)) * tauiceorig)
                scatliq = F(ssaliq * tauliq)
                scatice = F(ssaice * tauice)
                scatsno = F(0.0)
                taucmc[ig, lay] = F(tauliq + tauice)
            else:
                tauliqorig = F(clwpmc[ig, lay] * extcoliq)
                tauiceorig = F(ciwpmc[ig, lay] * extcoice)
                tausnoorig = F(cswpmc[ig, lay] * extcosno)
                taormc[ig, lay] = F(F(tauliqorig + tauiceorig) + tausnoorig)
                ssaliq = F(F(ssacoliq * F(one - forwliq)) /
                           F(one - F(forwliq * ssacoliq)))
                tauliq = F(F(one - F(forwliq * ssacoliq)) * tauliqorig)
                ssaice = F(F(ssacoice * F(one - forwice)) /
                           F(one - F(forwice * ssacoice)))
                tauice = F(F(one - F(forwice * ssacoice)) * tauiceorig)
                ssasno = F(F(ssacosno * F(one - forwsno)) /
                           F(one - F(forwsno * ssacosno)))
                tausno = F(F(one - F(forwsno * ssacosno)) * tausnoorig)
                scatliq = F(ssaliq * tauliq)
                scatice = F(ssaice * tauice)
                scatsno = F(ssasno * tausno)
                taucmc[ig, lay] = F(F(tauliq + tauice) + tausno)

            if taucmc[ig, lay] == F(0.0):
                taucmc[ig, lay] = cldmin
            if scatice == F(0.0):
                scatice = cldmin
            if scatsno == F(0.0):
                scatsno = cldmin

            if iceflag < 5:
                ssacmc[ig, lay] = F(F(scatliq + scatice) / taucmc[ig, lay])
            else:
                ssacmc[ig, lay] = F(F(F(scatliq + scatice) + scatsno) /
                                    taucmc[ig, lay])

            if iceflag == 3 or iceflag == 4:
                # istr = 1; x**istr with integer power 1 is x exactly
                asmcmc[ig, lay] = F(F(one / F(scatliq + scatice)) *
                    F(F(F(scatliq * F(gliq - forwliq)) / F(one - forwliq)) +
                      F(scatice * F(F(gice - forwice) / F(one - forwice)))))
            elif iceflag == 5:
                asmcmc[ig, lay] = F(F(one / F(F(scatliq + scatice) + scatsno)) *
                    F(F(F(F(scatliq * F(gliq - forwliq)) / F(one - forwliq)) +
                        F(scatice * F(F(gice - forwice) / F(one - forwice)))) +
                      F(scatsno * F(F(gsno - forwsno) / F(one - forwsno)))))
            else:
                asmcmc[ig, lay] = F(
                    F(F(F(scatliq * F(gliq - forwliq)) / F(one - forwliq)) +
                      F(F(scatice * F(gice - forwice)) / F(one - forwice))) /
                    F(scatliq + scatice))

    return taormc


# ===========================================================================
# SECTION 6: reftra_sw and vrtqdr_sw (two-stream layer + vertical quadrature)
# ===========================================================================

def _exp_tbl_lookup(tab: SWTables, ze1):
    """The shared exponential-lookup idiom of reftra_sw / spcvmc_sw."""
    if ze1 <= tab.od_lo:
        return F(F(F(1.0) - ze1) + F(F(F(0.5) * ze1) * ze1))
    tblind = F(ze1 / F(tab.bpade + ze1))
    itind = int(F(F(tab.tblint * tblind) + F(0.5)))
    return tab.exp_tbl[itind]


def reftra_sw(tab: SWTables, nlayers, lrtchk, pgg, prmuz, ptau, pw,
              pref, prefd, ptra, ptrad):
    """Port of reftra_sw (kmodts = 2, PIFM).  Writes pref/prefd/ptra/ptrad
    rows 0..nlayers-1 in place, exactly like the Fortran inout dummies."""
    one = F(1.0)
    eps = F(1.0e-08)
    zwcrit = F(0.9999995)

    for jk in range(nlayers):
        if not lrtchk[jk]:
            pref[jk] = F(0.0)
            ptra[jk] = one
            prefd[jk] = F(0.0)
            ptrad[jk] = one
            continue
        zto1 = ptau[jk]
        zw = pw[jk]
        zg = pgg[jk]

        zg3 = F(F(3.0) * zg)
        # kmodts == 2
        zgamma1 = F(F(F(8.0) - F(zw * F(F(5.0) + zg3))) * F(0.25))
        zgamma2 = F(F(F(3.0) * F(zw * F(one - zg))) * F(0.25))
        zgamma3 = F(F(F(2.0) - F(zg3 * prmuz)) * F(0.25))
        zgamma4 = F(one - zgamma3)

        zwo = F(0.0)
        denom = one
        if zg != one:
            q = F(zg / F(one - zg))
            denom = F(one - F(F(one - zw) * F(q * q)))
        if zw > F(0.0) and denom != F(0.0):
            zwo = F(zw / denom)

        if zwo >= zwcrit:
            # conservative scattering
            za = F(zgamma1 * prmuz)
            za1 = F(za - zgamma3)
            zgt = F(zgamma1 * zto1)
            ze1 = min(F(zto1 / prmuz), F(500.0))
            ze2 = _exp_tbl_lookup(tab, ze1)
            pref[jk] = F(F(zgt - F(za1 * F(one - ze2))) / F(one + zgt))
            ptra[jk] = F(one - pref[jk])
            prefd[jk] = F(zgt / F(one + zgt))
            ptrad[jk] = F(one - prefd[jk])
            if ze2 == one:
                pref[jk] = F(0.0)
                ptra[jk] = one
                prefd[jk] = F(0.0)
                ptrad[jk] = one
        else:
            # non-conservative scattering
            za1 = F(F(zgamma1 * zgamma4) + F(zgamma2 * zgamma3))
            za2 = F(F(zgamma1 * zgamma3) + F(zgamma2 * zgamma4))
            zrk = F(np.sqrt(F(F(zgamma1 * zgamma1) - F(zgamma2 * zgamma2)),
                            dtype=np.float32))
            zrp = F(zrk * prmuz)
            zrp1 = F(one + zrp)
            zrm1 = F(one - zrp)
            zrk2 = F(F(2.0) * zrk)
            zrpp = F(one - F(zrp * zrp))
            zrkg = F(zrk + zgamma1)
            zr1 = F(zrm1 * F(za2 + F(zrk * zgamma3)))
            zr2 = F(zrp1 * F(za2 - F(zrk * zgamma3)))
            zr3 = F(zrk2 * F(zgamma3 - F(za2 * prmuz)))
            zr4 = F(zrpp * zrkg)
            zr5 = F(zrpp * F(zrk - zgamma1))
            zt1 = F(zrp1 * F(za1 + F(zrk * zgamma4)))
            zt2 = F(zrm1 * F(za1 - F(zrk * zgamma4)))
            zt3 = F(zrk2 * F(zgamma4 + F(za1 * prmuz)))
            zt4 = zr4
            zt5 = zr5
            zbeta = F(F(zgamma1 - zrk) / zrkg)

            ze1 = min(F(zrk * zto1), F(500.0))
            ze2 = min(F(zto1 / prmuz), F(500.0))
            if ze1 <= tab.od_lo:
                zem1 = F(F(F(1.0) - ze1) + F(F(F(0.5) * ze1) * ze1))
                zep1 = F(one / zem1)
            else:
                tblind = F(ze1 / F(tab.bpade + ze1))
                itind = int(F(F(tab.tblint * tblind) + F(0.5)))
                zem1 = tab.exp_tbl[itind]
                zep1 = F(one / zem1)
            if ze2 <= tab.od_lo:
                zem2 = F(F(F(1.0) - ze2) + F(F(F(0.5) * ze2) * ze2))
                zep2 = F(one / zem2)
            else:
                tblind = F(ze2 / F(tab.bpade + ze2))
                itind = int(F(F(tab.tblint * tblind) + F(0.5)))
                zem2 = tab.exp_tbl[itind]
                zep2 = F(one / zem2)

            zdenr = F(F(zr4 * zep1) + F(zr5 * zem1))
            zdent = F(F(zt4 * zep1) + F(zt5 * zem1))
            if -eps <= zdenr <= eps:
                pref[jk] = eps
                ptra[jk] = zem2
            else:
                pref[jk] = F(F(zw * F(F(F(zr1 * zep1) - F(zr2 * zem1)) -
                                      F(zr3 * zem2))) / zdenr)
                # Fortran: zem2 - zem2 * zw * (...) / zdent
                #        = zem2 - (((zem2*zw) * (...)) / zdent)
                ptra[jk] = F(zem2 - F(F(F(zem2 * zw) * F(F(F(zt1 * zep1) -
                             F(zt2 * zem1)) - F(zt3 * zep2))) / zdent))

            zemm = F(zem1 * zem1)
            zdend = F(one / F(F(one - F(zbeta * zemm)) * zrkg))
            prefd[jk] = F(F(zgamma2 * F(one - zemm)) * zdend)
            ptrad[jk] = F(F(zrk2 * zem1) * zdend)


def vrtqdr_sw(klev, kw, pref, prefd, ptra, ptrad, pdbt, prdnd, prup, prupd,
              ptdbt, pfd, pfu):
    """Port of vrtqdr_sw.  0-based arrays; kw is the Fortran g-point index
    (1-based) -> column kw-1 of pfd/pfu.  prdnd/prup/prupd updated in place."""
    one = F(1.0)
    ztdn = np.zeros(klev + 1, dtype=np.float32)

    zreflect = F(one / F(one - F(prefd[klev] * prefd[klev - 1])))
    prup[klev - 1] = F(pref[klev - 1] + F(F(ptrad[klev - 1] *
        F(F(F(ptra[klev - 1] - pdbt[klev - 1]) * prefd[klev]) +
          F(pdbt[klev - 1] * pref[klev]))) * zreflect))
    prupd[klev - 1] = F(prefd[klev - 1] + F(F(F(ptrad[klev - 1] *
        ptrad[klev - 1]) * prefd[klev]) * zreflect))

    for jk in range(1, klev):
        ikp = klev + 1 - jk   # Fortran level index
        ikx = ikp - 1
        zreflect = F(one / F(one - F(prupd[ikp - 1] * prefd[ikx - 1])))
        prup[ikx - 1] = F(pref[ikx - 1] + F(F(ptrad[ikx - 1] *
            F(F(F(ptra[ikx - 1] - pdbt[ikx - 1]) * prupd[ikp - 1]) +
              F(pdbt[ikx - 1] * prup[ikp - 1]))) * zreflect))
        prupd[ikx - 1] = F(prefd[ikx - 1] + F(F(F(ptrad[ikx - 1] *
            ptrad[ikx - 1]) * prupd[ikp - 1]) * zreflect))

    ztdn[0] = one
    prdnd[0] = F(0.0)
    ztdn[1] = ptra[0]
    prdnd[1] = prefd[0]

    for jk in range(2, klev + 1):
        ikp = jk + 1
        zreflect = F(one / F(one - F(prefd[jk - 1] * prdnd[jk - 1])))
        ztdn[ikp - 1] = F(F(ptdbt[jk - 1] * ptra[jk - 1]) +
            F(F(ptrad[jk - 1] * F(F(ztdn[jk - 1] - ptdbt[jk - 1]) +
                F(F(ptdbt[jk - 1] * pref[jk - 1]) * prdnd[jk - 1]))) *
              zreflect))
        prdnd[ikp - 1] = F(prefd[jk - 1] + F(F(F(ptrad[jk - 1] *
            ptrad[jk - 1]) * prdnd[jk - 1]) * zreflect))

    for jk in range(klev + 1):
        zreflect = F(one / F(one - F(prdnd[jk] * prupd[jk])))
        pfu[jk, kw - 1] = F(F(F(ptdbt[jk] * prup[jk]) +
            F(F(ztdn[jk] - ptdbt[jk]) * prupd[jk])) * zreflect)
        pfd[jk, kw - 1] = F(ptdbt[jk] + F(F(F(ztdn[jk] - ptdbt[jk]) +
            F(F(ptdbt[jk] * prup[jk]) * prdnd[jk])) * zreflect))


# ===========================================================================
# SECTION 7: spcvmc_sw (spectral loop, two-stream McICA)
# ===========================================================================

def spcvmc_sw(tab: SWTables, nlayers, istart, iend, icpr,
              palbd, palbp, pcldfmc, ptaucmc, pasycmc, pomgcmc, ptaormc,
              ptaua, pasya, pomga, prmu0,
              adjflux, laytrop, jp, jt, jt1,
              colch4, colco2, colh2o, colmol, colo2, colo3,
              fac00, fac01, fac10, fac11,
              selffac, selffrac, indself, forfac, forfrac, indfor):
    """Port of spcvmc_sw (iout = 0 path; icpr must be >= 1, as rrtmg_sw
    always calls it after cldprmc.  The icpr == 0 delta-scale-here branch
    is unreachable via WRF option 4 and fails closed).

    pcldfmc/ptaucmc/... are (nlayers, ngptsw); ptaua/pasya/pomga are
    (nlayers, nbndsw); adjflux is indexed by Fortran band number 16..29 as
    adjflux[jb - 16].  Returns a dict of the 14 pbb*/puv*/pni* level arrays
    plus the g-point flux arrays for the CUDA twin's gates.
    """
    if icpr < 1:
        raise NotImplementedError(
            "spcvmc_sw icpr == 0 branch is unreachable via WRF option 4")
    one = F(1.0)
    klev = nlayers
    repclc = F(1.0e-12)

    pbbcd = np.zeros(klev + 1, dtype=np.float32)
    pbbcu = np.zeros(klev + 1, dtype=np.float32)
    pbbfd = np.zeros(klev + 1, dtype=np.float32)
    pbbfu = np.zeros(klev + 1, dtype=np.float32)
    pbbcddir = np.zeros(klev + 1, dtype=np.float32)
    pbbfddir = np.zeros(klev + 1, dtype=np.float32)
    puvcd = np.zeros(klev + 1, dtype=np.float32)
    puvfd = np.zeros(klev + 1, dtype=np.float32)
    puvcddir = np.zeros(klev + 1, dtype=np.float32)
    puvfddir = np.zeros(klev + 1, dtype=np.float32)
    pnicd = np.zeros(klev + 1, dtype=np.float32)
    pnifd = np.zeros(klev + 1, dtype=np.float32)
    pnicddir = np.zeros(klev + 1, dtype=np.float32)
    pnifddir = np.zeros(klev + 1, dtype=np.float32)

    zsflxzen, ztaug, ztaur = taumol_sw(
        tab, klev, colh2o, colco2, colch4, colo2, colo3, colmol,
        laytrop, jp, jt, jt1, fac00, fac01, fac10, fac11,
        selffac, selffrac, indself, forfac, forfrac, indfor)

    zcd = np.zeros((klev + 1, NGPTSW), dtype=np.float32)
    zcu = np.zeros((klev + 1, NGPTSW), dtype=np.float32)
    zfd = np.zeros((klev + 1, NGPTSW), dtype=np.float32)
    zfu = np.zeros((klev + 1, NGPTSW), dtype=np.float32)
    zincflx = np.zeros(NGPTSW, dtype=np.float32)

    lrtchkclr = np.ones(klev, dtype=bool)
    lrtchkcld = np.zeros(klev, dtype=bool)
    ztauc = np.zeros(klev, dtype=np.float32)
    zomcc = np.zeros(klev, dtype=np.float32)
    zgcc = np.zeros(klev, dtype=np.float32)
    ztauo = np.zeros(klev, dtype=np.float32)
    zomco = np.zeros(klev, dtype=np.float32)
    zgco = np.zeros(klev, dtype=np.float32)
    zrefc = np.zeros(klev + 1, dtype=np.float32)
    zrefdc = np.zeros(klev + 1, dtype=np.float32)
    ztrac = np.zeros(klev + 1, dtype=np.float32)
    ztradc = np.zeros(klev + 1, dtype=np.float32)
    zrefo = np.zeros(klev + 1, dtype=np.float32)
    zrefdo = np.zeros(klev + 1, dtype=np.float32)
    ztrao = np.zeros(klev + 1, dtype=np.float32)
    ztrado = np.zeros(klev + 1, dtype=np.float32)
    zref = np.zeros(klev + 1, dtype=np.float32)
    zrefd = np.zeros(klev + 1, dtype=np.float32)
    ztra = np.zeros(klev + 1, dtype=np.float32)
    ztrad = np.zeros(klev + 1, dtype=np.float32)
    zdbtc = np.zeros(klev + 1, dtype=np.float32)
    ztdbtc = np.zeros(klev + 1, dtype=np.float32)
    zdbt = np.zeros(klev + 1, dtype=np.float32)
    ztdbt = np.zeros(klev + 1, dtype=np.float32)
    zdbtc_nodel = np.zeros(klev + 1, dtype=np.float32)
    ztdbtc_nodel = np.zeros(klev + 1, dtype=np.float32)
    zdbt_nodel = np.zeros(klev + 1, dtype=np.float32)
    ztdbt_nodel = np.zeros(klev + 1, dtype=np.float32)
    zrdnd = np.zeros(klev + 1, dtype=np.float32)
    zrdndc = np.zeros(klev + 1, dtype=np.float32)
    zrup = np.zeros(klev + 1, dtype=np.float32)
    zrupd = np.zeros(klev + 1, dtype=np.float32)
    zrupc = np.zeros(klev + 1, dtype=np.float32)
    zrupdc = np.zeros(klev + 1, dtype=np.float32)

    iw = 0
    for jb in range(istart, iend + 1):
        ibm = jb - 15          # 1..14
        igt = NGC[ibm - 1]
        for _jg in range(1, igt + 1):
            iw += 1
            zincflx[iw - 1] = F(F(adjflux[jb - 16] * zsflxzen[iw - 1]) * prmu0)

            ztdbtc[0] = one
            ztdbtc_nodel[0] = one
            zdbtc[klev] = F(0.0)
            ztrac[klev] = F(0.0)
            ztradc[klev] = F(0.0)
            zrefc[klev] = palbp[ibm - 1]
            zrefdc[klev] = palbd[ibm - 1]
            zrupc[klev] = palbp[ibm - 1]
            zrupdc[klev] = palbd[ibm - 1]
            ztdbt[0] = one
            ztdbt_nodel[0] = one
            zdbt[klev] = F(0.0)
            ztra[klev] = F(0.0)
            ztrad[klev] = F(0.0)
            zref[klev] = palbp[ibm - 1]
            zrefd[klev] = palbd[ibm - 1]
            zrup[klev] = palbp[ibm - 1]
            zrupd[klev] = palbd[ibm - 1]

            for jk in range(1, klev + 1):
                ikl = klev + 1 - jk
                J, K = jk - 1, ikl - 1
                lrtchkclr[J] = True
                lrtchkcld[J] = pcldfmc[K, iw - 1] > repclc

                ztauc[J] = F(F(ztaur[K, iw - 1] + ztaug[K, iw - 1]) +
                             ptaua[K, ibm - 1])
                zomcc[J] = F(F(ztaur[K, iw - 1] * one) +
                             F(ptaua[K, ibm - 1] * pomga[K, ibm - 1]))
                zgcc[J] = F(F(F(pasya[K, ibm - 1] * pomga[K, ibm - 1]) *
                              ptaua[K, ibm - 1]) / zomcc[J])
                zomcc[J] = F(zomcc[J] / ztauc[J])

                zclear = F(one - pcldfmc[K, iw - 1])
                zcloud = pcldfmc[K, iw - 1]

                ze1 = F(ztauc[J] / prmu0)
                zdbtmc = _exp_tbl_lookup(tab, ze1)
                zdbtc_nodel[J] = zdbtmc
                ztdbtc_nodel[J + 1] = F(zdbtc_nodel[J] * ztdbtc_nodel[J])

                tauorig = F(ztauc[J] + ptaormc[K, iw - 1])
                ze1 = F(tauorig / prmu0)
                zdbtmo = _exp_tbl_lookup(tab, ze1)
                zdbt_nodel[J] = F(F(zclear * zdbtmc) + F(zcloud * zdbtmo))
                ztdbt_nodel[J + 1] = F(zdbt_nodel[J] * ztdbt_nodel[J])

            for jk in range(1, klev + 1):
                J = jk - 1
                zf = F(zgcc[J] * zgcc[J])
                zwf = F(zomcc[J] * zf)
                ztauc[J] = F(F(one - zwf) * ztauc[J])
                zomcc[J] = F(F(zomcc[J] - zwf) / F(one - zwf))
                zgcc[J] = F(F(zgcc[J] - zf) / F(one - zf))

            # icpr >= 1
            for jk in range(1, klev + 1):
                ikl = klev + 1 - jk
                J, K = jk - 1, ikl - 1
                ztauo[J] = F(ztauc[J] + ptaucmc[K, iw - 1])
                zomco[J] = F(F(ztauc[J] * zomcc[J]) +
                             F(ptaucmc[K, iw - 1] * pomgcmc[K, iw - 1]))
                zgco[J] = F(F(F(F(ptaucmc[K, iw - 1] * pomgcmc[K, iw - 1]) *
                                pasycmc[K, iw - 1]) +
                              F(F(ztauc[J] * zomcc[J]) * zgcc[J])) / zomco[J])
                zomco[J] = F(zomco[J] / ztauo[J])

            reftra_sw(tab, klev, lrtchkclr, zgcc, prmu0, ztauc, zomcc,
                      zrefc, zrefdc, ztrac, ztradc)
            reftra_sw(tab, klev, lrtchkcld, zgco, prmu0, ztauo, zomco,
                      zrefo, zrefdo, ztrao, ztrado)

            for jk in range(1, klev + 1):
                ikl = klev + 1 - jk
                J, K = jk - 1, ikl - 1
                zclear = F(one - pcldfmc[K, iw - 1])
                zcloud = pcldfmc[K, iw - 1]
                zref[J] = F(F(zclear * zrefc[J]) + F(zcloud * zrefo[J]))
                zrefd[J] = F(F(zclear * zrefdc[J]) + F(zcloud * zrefdo[J]))
                ztra[J] = F(F(zclear * ztrac[J]) + F(zcloud * ztrao[J]))
                ztrad[J] = F(F(zclear * ztradc[J]) + F(zcloud * ztrado[J]))

                ze1 = F(ztauc[J] / prmu0)
                zdbtmc = _exp_tbl_lookup(tab, ze1)
                zdbtc[J] = zdbtmc
                ztdbtc[J + 1] = F(zdbtc[J] * ztdbtc[J])

                ze1 = F(ztauo[J] / prmu0)
                zdbtmo = _exp_tbl_lookup(tab, ze1)
                zdbt[J] = F(F(zclear * zdbtmc) + F(zcloud * zdbtmo))
                ztdbt[J + 1] = F(zdbt[J] * ztdbt[J])

            vrtqdr_sw(klev, iw, zrefc, zrefdc, ztrac, ztradc,
                      zdbtc, zrdndc, zrupc, zrupdc, ztdbtc, zcd, zcu)
            vrtqdr_sw(klev, iw, zref, zrefd, ztra, ztrad,
                      zdbt, zrdnd, zrup, zrupd, ztdbt, zfd, zfu)

            for jk in range(1, klev + 2):
                ikl = klev + 2 - jk
                J, K = jk - 1, ikl - 1
                pbbfu[K] = F(pbbfu[K] + F(zincflx[iw - 1] * zfu[J, iw - 1]))
                pbbfd[K] = F(pbbfd[K] + F(zincflx[iw - 1] * zfd[J, iw - 1]))
                pbbcu[K] = F(pbbcu[K] + F(zincflx[iw - 1] * zcu[J, iw - 1]))
                pbbcd[K] = F(pbbcd[K] + F(zincflx[iw - 1] * zcd[J, iw - 1]))
                pbbfddir[K] = F(pbbfddir[K] +
                                F(zincflx[iw - 1] * ztdbt_nodel[J]))
                pbbcddir[K] = F(pbbcddir[K] +
                                F(zincflx[iw - 1] * ztdbtc_nodel[J]))
            if 10 <= ibm <= 13:
                for jk in range(1, klev + 2):
                    ikl = klev + 2 - jk
                    J, K = jk - 1, ikl - 1
                    puvcd[K] = F(puvcd[K] + F(zincflx[iw - 1] * zcd[J, iw - 1]))
                    puvfd[K] = F(puvfd[K] + F(zincflx[iw - 1] * zfd[J, iw - 1]))
                    puvcddir[K] = F(puvcddir[K] +
                                    F(zincflx[iw - 1] * ztdbtc_nodel[J]))
                    puvfddir[K] = F(puvfddir[K] +
                                    F(zincflx[iw - 1] * ztdbt_nodel[J]))
            elif ibm == 14 or ibm <= 9:
                for jk in range(1, klev + 2):
                    ikl = klev + 2 - jk
                    J, K = jk - 1, ikl - 1
                    pnicd[K] = F(pnicd[K] + F(zincflx[iw - 1] * zcd[J, iw - 1]))
                    pnifd[K] = F(pnifd[K] + F(zincflx[iw - 1] * zfd[J, iw - 1]))
                    pnicddir[K] = F(pnicddir[K] +
                                    F(zincflx[iw - 1] * ztdbtc_nodel[J]))
                    pnifddir[K] = F(pnifddir[K] +
                                    F(zincflx[iw - 1] * ztdbt_nodel[J]))

    return dict(pbbfd=pbbfd, pbbfu=pbbfu, pbbcd=pbbcd, pbbcu=pbbcu,
                puvfd=puvfd, puvcd=puvcd, pnifd=pnifd, pnicd=pnicd,
                pbbfddir=pbbfddir, pbbcddir=pbbcddir, puvfddir=puvfddir,
                puvcddir=puvcddir, pnifddir=pnifddir, pnicddir=pnicddir,
                zincflx=zincflx, zcd=zcd, zcu=zcu, zfd=zfd, zfu=zfu,
                zsflxzen=zsflxzen, ztaug=ztaug, ztaur=ztaur)


# --- BEGIN GENERATED STATIC TABLES (tools/rrtmg_wrf461_oracle/sw_gen_static.py) ---
# Source: sw_tables.npz oracle dump of the unmodified WRF modules post rrtmg_sw_ini.
_STATIC_TABLES = {
    "cld/extliq1": ("f4", (58, 14),
        "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"),
    "cld/ssaliq1": ("f4", (58, 14),
        "/BFWPxFSTz/ztkY/hqQ/Pxa7Oj9UUjc/t8A0P4+lMj/70jA/5TEvP462LT/JWSw/"
        "9hYrP/npKT9z0Cg/g8gnPwPQJj9D5SU/UQclP8w0JD/DbCM/fa4iPxP5IT8KTCE/"
        "TqYgPykIID+2cB8/Rt8ePy5UHj+Ozh0/Lk4dPxvTHD9qXRw/RuwbP+F/Gz81GBs/"
        "xrQaP5BVGj/U+hk/QqQZPztRGT/6ARk/yLYYP55uGD9qKhg/MOkXP56rFz9XcBc/"
        "RzgXPx4DFz+n0BY/LKAWP4hyFj+rRhY/ch0WP9H1FT9l0BU/JKwVP3EVMz+/Pjc/"
        "SCU1P2NWMT+Tuy0/v9cqPxaZKD9R0SY/xVklPw0aJD+IAiM/XQkiP7EnIT8ZWSA/"
        "LJofP/7nHj/AQB4/9aIdP40NHT9Kfxw/xvcbP/p1Gz+k+Ro/TIIaP5IPGj8IoRk/"
        "+jYZP13QGD/QbRg/aA4YP8ayFz+fWhc/dQUXP4uzFj+8ZBY/+BgWPyzQFT+YihU/"
        "rEcVP50HFT8JyhQ/rY8UP5hXFD8IIhQ/v+4TP1++Ez80kBM/smMTP5A5Ez9mERM/"
        "MOsSP/PGEj8rpBI/qIMSP/9jEj8jRhI/lykSP3MOEj9bM34/mVx9P744fD/6DXs/"
        "tfh5Pzj4eD+3B3g/WiN3P0BJdj9Wd3U/3qt0P+Llcz9UJHM/cGZyP8GrcT/+83A/"
        "3T5wP+uLbz+C224/PS1uPwmBbT/v1mw/5y5sP7+Iaz+y5Go/gkJqP3aiaT+SBGk/"
        "wmhoPznPZz/5N2c/Y6NmP0URZj8vgmU/6/VkP+psZD/35mM/sGRjP8DlYj98amI/"
        "A/NhP2F/YT9nD2E/cqNgPxs7YD9i1l8//XVfP70YXz9Nv14/u2lePykXXj/Fx10/"
        "33tdP0czXT8y7Vw/eapcP3dqXD/FLFw/MLZ9P6hEfD+QwHo/M2t5P643eD8lGXc/"
        "WAh2P14BdT/3AHQ/4wVzP/8Ocj8tHHE/PC1wPw1Cbz+5Wm4/6XZtP8eWbD/MuWs/"
        "+99qP2QJaj+tNWk/5WRoP7SWZz9iy2Y/rgJmP688ZT86eWQ/rbhjP876Yj/TP2I/"
        "CIhhP3fTYD9IImA/fXRfP6/KXj9+JF4/jYJdP+fkXD+yS1w/4rZbP7kmWz9/m1o/"
        "sRRaP32SWT8nFVk/h5xYP0YoWD+nuFc/R01XPyrmVj9rg1Y/piRWP+HJVT+9clU/"
        "UR9VP0DPVD+LglQ/PzlUPxg6fz89u34/vT5+PyzOfT/OZ30/bAl9PzWwfD/+WXw/"
        "TAV8P2Sxez8+Xns/ugt7P/W5ej8gaXo/JBl6PwHKeT/Xe3k/cy55P8vheD8Qlng/"
        "8kp4P4sAeD/Ttnc/zW13P1Ildz+63XY/upZ2P0hQdj+rCnY/9MV1P/iBdT/6PnU/"
        "EP10PxO8dD9yfHQ/6z10P90AdD83xXM/FItzP3lScz9YG3M/AOZyP0Wycj8+gHI/"
        "AlByP1Uhcj9U9HE//8hxP1GfcT9Qd3E/zFBxP+ArcT8qCHE/C+ZwP13FcD/6pXA/"
        "xIdwP+lqcD8nfX8/CTJ/P0/xfj+ptn4/635+PzFJfj9KFH4/6N99P8arfT9/d30/"
        "VkN9P0sPfT8v23w/Vad8P4lzfD8MQHw/4Qx8P/XZez9Pp3s/LXV7PyhDez99EXs/"
        "EeB6Pwevej/2fXo/U016P/Icej/M7Hk/GL15P32NeT9yXnk/6S95P6QBeT8V1Hg/"
        "L6d4P/V6eD+bT3g//SR4P2D7dz/B0nc/Eat3P4OEdz/0Xnc/rDp3P3kXdz9m9XY/"
        "ZdR2P6+0dj/7lXY/YHh2Pwhcdj+RQHY/IyZ2P+IMdj+P9HU/Jt11P6PGdT8CsXU/"
        "y7J/P4KIfz/RY38/YUJ/P3Uifz8eBH8/KuZ+P5TIfj8Yq34/sI1+P+9vfj/gUX4/"
        "GjR+PwwWfj/+930/INp9P1m8fT/0nn0/pIF9P7NkfT/zR30/hSt9PyMPfT8O83w/"
        "Etd8P0+7fD+4n3w/ToR8Px9pfD/9TXw/GzN8P4QYfD9W/ns/SeR7P6HKez9/sXs/"
        "nph7P3uAez+iaHs/gVF7P7M6ez+5JHs/SQ97P5r6ej9o5no/ANN6Pz3Aej8Grno/"
        "h5x6P7GLej9we3o/6Gt6P9Ncej8sTno/OkB6P94yej/6JXo/lhl6P874fz979X8/"
        "VvJ/Pzzvfz9y7H8/uOl/Pxnnfz+E5H8/7eF/P0Tffz+K3H8/2Nl/PyXXfz961H8/"
        "ydF/PxDPfz9czH8/wcl/PxLHfz9nxH8/zsF/PzS/fz+avH8/B7p/P2i3fz/YtH8/"
        "RrJ/P8qvfz8/rX8/u6p/Pymofz/RpX8/V6N/P+Cgfz+Bnn8/I5x/P92Zfz+Sl38/"
        "UpV/PzKTfz8akX8/AY9/Pw6Nfz8Qi38/Mol/P2KHfz+bhX8/7YN/Pz6Cfz+7gH8/"
        "L39/P6h9fz9TfH8/5Xp/P5l5fz9yeH8/Ind/Pwl2fz/b/38/qf9/P3H/fz9h/38/"
        "Uv9/P0L/fz9E/38/RP9/Pyb/fz8L/38/Df9/Pw7/fz/y/n8/6f5/P+H+fz/A/n8/"
        "y/5/P6X+fz+S/n8/jf5/P4H+fz90/n8/Z/5/P1P+fz9T/n8/Rf5/Pzn+fz8k/n8/"
        "E/5/P/v9fz/4/X8/3/1/P9v9fz/G/X8/vP1/P7D9fz+w/X8/i/1/P4L9fz94/X8/"
        "d/1/P239fz9o/X8/V/1/P079fz9J/X8/Kv1/Pw39fz8Z/X8/AP1/Pwb9fz/7/H8/"
        "9vx/P+X8fz/g/H8/x/x/P/L8fz/F/H8/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAP/j/fz/4/38/7/9/P/H/fz/x/38/7/9/P+7/fz/s/38/7P9/P+n/fz/s/38/"
        "6f9/P+r/fz/p/38/5/9/P+f/fz/n/38/5/9/P+P/fz/j/38/4P9/P+D/fz/p/38/"
        "5/9/P+P/fz/l/38/4v9/P+P/fz/i/38/3v9/P+D/fz/b/38/3f9/P+f/fz/n/38/"
        "4/9/P+P/fz/j/38/4/9/PwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/"
        "AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD8AAIA/AACAPwAAgD/x/38/"
        "7/9/P/P/fz/4/38/6f9/P/b/fz/n/38/3v9/P+X/fz/j/38/3f9/P8z/fz/Z/38/"
        "z/9/P8z/fz/M/38/xf9/P87/fz/n/38/0/9/P7//fz/K/38/r/9/P87/fz/U/38/"
        "xf9/P8z/fz/O/38/zv9/P73/fz9//38/uv9/P5b/fz+9/38/sf9/P6X/fz+l/38/"
        "uP9/P67/fz8AAIA/AACAP+P/fz/d/38/xf9/P6r/fz+p/38/dv9/P3X/fz9//38/"
        "Zv9/P0b/fz9c/38/UP9/Pzj/fz86/38/MP9/Px//fz8f/38/Ev9/PxX/fz8I/38/"
        "7f5/P+T+fz/P/n8/vv5/P8v+fz+w/n8/lP5/P4/+fz9y/n8/Vv5/Pz3+fz8e/n8/"
        "cf5/P0z+fz/W/X8/Dv5/P/P9fz/Q/X8/7P1/P9H9fz9z/X8/nP1/P2H9fz+c/X8/"
        "V/1/P2P9fz9J/X8/HP1/Py39fz8a/X8/4Px/P/T8fz/q/H8/Ev1/P7H8fz+o/H8/"
        "ev9/Pyv/fz8N/38/5v5/P7z+fz+M/n8/Y/5/Pyz+fz/7/X8/tv1/P4T9fz9t/X8/"
        "Kv1/Pwb9fz/c/H8/qvx/P3H8fz9T/H8/Fvx/P+b7fz/D+38/g/t/P0v7fz8M+38/"
        "+/p/P6/6fz9M+n8/N/p/P+35fz+U+X8/Z/l/P1z5fz/v+H8/v/h/P5b4fz8r+H8/"
        "/vd/P9H3fz+I938/U/d/P/v2fz8a938/tfZ/P3r2fz959n8/BfZ/Pxb2fz/y9X8/"
        "ifV/P5j1fz+O9X8/IvV/Pwf1fz8T9X8/C/V/P8v0fz+09H8/hvR/P8OfVj8ZnVg/"
        "bFtWP/0/Uj8AaU0/a29IPyahQz8tIT8/pfs6P4wxNz+GvjM/PZwwP1DDLT+CLCs/"
        "J9EoP6uqJj+wsyQ/weYiP70/IT+Fuh8/nFMePyUIHT9C1Rs/jrgaPyOwGT8wuhg/"
        "3NQXP/j+Fj8zNxY/M3wVP0zNFD+QKRQ/BZATPwMAEz8ReRI/gPoRP9uDET9oFBE/"
        "8qsQP79JED/V7Q8/oZcPP55GDz9w+g4/OrMOPz5wDj9jMQ4/UPYNP7i+DT+jig0/"
        "q1kNP4wrDT8JAA0/cNcMP/KwDD+GjAw/XWoMPwhKDD8="),
    "cld/asyliq1": ("f4", (58, 14),
        "68ZNP48pTT8wU0s/mwtMP3sETz9cxlI/rmBWP7J9WT/wFFw/gThePwgAYD/4f2E/"
        "s8hiP8rmYz9W42Q/C8VlP6yQZj8USmc/7/NnP5SQaD+BIWk/AahpP2claj92mmo/"
        "aAhrPzhvaz/4z2s/RCtsPw+BbD9Q0mw//R5tP5lnbT8nrG0/LO1tP7sqbj/+ZG4/"
        "PZxuP6zQbj8wAm8/FTFvP5hdbz+5h28/k69vP17Vbz8o+W8/9BpwP+Y6cD8uWXA/"
        "4HVwPwuRcD/OqnA/JMNwPzracD8h8HA/1QRxP2kYcT8PK3E/9TxxP73jZD+eyGc/"
        "3qpnP8MYZz8y9GY/o15nPy8waD8ZNmk/okpqP4JXaz9xUWw/xTNtPz3+bT8Vsm4/"
        "zFFvPxfgbz84X3A/atFwP8k4cT/mlnE/+OxxPy88cj9zhXI/OslyP40Icz/aQ3M/"
        "SHtzP4evcz+T4HM/GQ90P/o6dD9oZHQ/mYt0PwCxdD8H1HQ/hPV0P1gVdT9OM3U/"
        "xk91P8VqdT93hHU/q5x1P42zdT90yXU/9t11P3rxdT/1A3Y/fxV2PxImdj+rNXY/"
        "t0R2P8xSdj8NYHY/rGx2P7N4dj8WhHY/2I52PyqZdj+/gls/oMtVP9B1UD8y7U4/"
        "ExxQPz5TUj9RqlQ/dshWP5yWWD/oF1o/P1hbP7RkXD+cSF0/ig1eP/K5Xj+wU18/"
        "l95fP1NdYD8X0mA/dj5hP9ijYT/wAmI/pFxiP2qxYj8HAmM/fU5jP5OXYz813WM/"
        "5x9kP5FfZD+snGQ/OtdkP2cPZT9PRWU/EXllP6SqZT9E2mU/9gdmP5AzZj9uXWY/"
        "jYVmP+KrZj+E0GY/lPNmPwQVZz8RNWc/mlNnP9hwZz+ejGc/FqdnP1zAZz+O2Gc/"
        "pO9nP4YFaD+BGmg/jy5oP51BaD/SU2g/vL9VP2uvTT8MEEs/XA9NPzBPUD8xVlM/"
        "Q89VP53DVz8lUVk/3ZNaPwGhWz/Ph1w/FlJdP8UGXj/eql4/60BfP6vLXz/MTGA/"
        "qMVgPyk3YT8vomE/XwdiP25nYj+fwmI/XxljP2BsYz+Vu2M/bwdkP/5PZD+jlWQ/"
        "fthkP48YZT8vVmU/UZFlP/bJZT+UAGY/2TRmPwRnZj8ll2Y/Z8VmP5HxZj/uG2c/"
        "TERnPw1rZz8PkGc/c7NnPz3VZz+P9Wc/XRRoP8MxaD/qTWg/smhoP1eCaD/Hmmg/"
        "HbJoP2rIaD+P3Wg/5fFoP62bTz/a30g/AcdJPx4LTT+vKFA/wLJSP8y4VD/FWFY/"
        "A61XP/nIWD8Fulk/vYlaP7U+Wz+u3Vs/kWpcP9DnXD8lWF0/c71dP2kZXj9+bV4/"
        "zrpeP2ACXz8gRV8/qYNfP4i+Xz8q9l8/BStgP1FdYD8tjWA/7rpgP8LmYD+jEGE/"
        "xThhPzNfYT/Mg2E/DKdhP9LIYT/46GE/xgdiPyIlYj86QWI/51tiP291Yj/DjWI/"
        "7qRiP+m6Yj/bz2I/7eNiP+r2Yj/wCGM/LhpjP4EqYz8kOmM/30hjP/9WYz9cZGM/"
        "J3FjP4N9Yz92qkg/u8JHP9tMTD+RfVA/bYBTP9CcVT9HJlc/fFRYP0pIWT9sFFo/"
        "r8NaPxRdWz+K5Fs/R11cP43JXD85K10/8YNdP+vUXT8wH14/62NeP7WjXj80314/"
        "AhdfP3lLXz8FfV8//KtfP4/YXz/bAmA/PitgP8xRYD+MdmA/iJlgP/26YD8d22A/"
        "n/lgP7kWYT+bMmE/OE1hP5ZmYT/NfmE/0pVhP8urYT+9wGE/nNRhP6TnYT+l+WE/"
        "1ApiPygbYj+jKmI/dTliP35HYj/eVGI/n2FiP7dtYj9IeWI/P4RiP8WOYj+zmGI/"
        "Ea1GP1D9ST/iG08/yLFSP+0GVT8To1Y/3thXPxjSWD9to1k//lZaP0f0Wj9Hfls/"
        "RPhbP5ZkXD+SxVw/jRxdP0trXT/Qsl0/V/RdP+kwXj//aF4/Up1eP1/OXj+f/F4/"
        "VihfP4hRXz/BeF8/HJ5fP5TBXz+Q418/rANgP5IiYD/GP2A/wFtgP252YD/Nj2A/"
        "9adgP+G+YD/h1GA/e+lgPwn9YD/bD2E/iiFhP3syYT98QmE/qFFhP/lfYT+KbWE/"
        "d3phP6GGYT8kkmE/L51hP6ynYT9lsWE/8rphP73DYT9CzGE/OdRhPwvoSD+eW08/"
        "4WNTP1XFVT+xWFc/YIFYP8prWT97LVo/8tFaP9ZfWz+B21s/9EdcP2unXD8K/Fw/"
        "hEddPy+LXT83yF0/vf9dP5AyXj80YV4/d4xeP5u0Xj8N2l4/Ef1eP+IdXz/UPF8/"
        "/llfP2N1Xz9Rj18/26dfPx2/Xz/61F8/uelfP3L9Xz8NEGA/tSFgP2IyYD8pQmA/"
        "GlFgPz1fYD+nbGA/SHlgPzCFYD+AkGA/QZtgP1KlYD/YrmA/6LdgP2rAYD90yGA/"
        "D9BgP03XYD8d3mA/qeRgP8fqYD+N8GA/EPZgPz37YD/RME4/6J1TP416Vj/tUFg/"
        "zKdZP5yuWj8ke1s//xtcP+OcXD/cBl0/fmBdP/GtXT9y8l0/oi9eP+hmXj8CmV4/"
        "ycZeP5/wXj/+Fl8/UTpfP+5aXz8reV8/G5VfPymvXz92x18/Jd5fP1rzXz9qB2A/"
        "SRpgP8ArYD9rPGA/BExgP9BaYD/BaGA/9XVgP1iCYD8DjmA/LZlgP6OjYD94rWA/"
        "zrZgP5y/YD/rx2A/xc9gPxfXYD8T3mA/ruRgP+jqYD/B8GA/T/ZgP4X7YD9xAGE/"
        "KAVhP3YJYT+fDWE/nRFhP2AVYT/vGGE/JTBTP7zVVj/o71g/oUlaP+s8Wz8V91s/"
        "NYxcP2gGXT83bF0/lsJdP+IMXj/ETV4/I4deP1i6Xj9Y6F4/xxFfP6s3Xz8JWl8/"
        "o3lfP6KWXz87sV8/5slfP7XgXz/09V8/6QlgP0ccYD+gLWA/7j1gPylNYD9kW2A/"
        "/2hgP8V1YD/IgWA/+YxgP7mXYD/NoWA/W6tgP3G0YD/yvGA/8MRgP3TMYD+w02A/"
        "Y9pgP97gYD/V5mA/j+xgP+/xYD/+9mA/u/tgP2QAYT+WBGE/lwhhP2UMYT8REGE/"
        "XxNhP6QWYT+9GWE/nhxhPzYSVT9hzVc/TY1ZPyfNWj8Jwls/XYFcPwMXXT/XjF0/"
        "B+tdP+U3Xj/td14/da5eP3TdXj+1Bl8/YCtfPxRMXz+uaV8/mYRfP0KdXz/cs18/"
        "4shfPx7cXz9G7l8/+f5fP3wOYD8UHWA/wSpgP4Y3YD9sQ2A/105gPyZZYD8bY2A/"
        "ZGxgP0V1YD9ffWA/G4VgP2SMYD8Lk2A/e5lgP52fYD9KpWA/tKpgP6avYD97tGA/"
        "67hgPyG9YD8nwWA/0MRgP0vIYD/My2A/5s5gP9rRYD/A1GA/QddgP8/ZYD9E3GA/"
        "hN5gP53gYD/28lY/qBdZPxmFWj+Ef1s/by5cP06uXD+dEV0/jWNdPwmqXT876F0/"
        "mh9eP+dQXj+jfF4/ZaNeP7vFXj8S5F4/Fv9ePzQXXz/ULF8/SkBfP+FRXz/8YV8/"
        "xHBfPz9+Xz/hil8/iJZfPz6hXz91q18/2rRfP7G9Xz8Yxl8/8s1fP07VXz8n3F8/"
        "luJfP9/oXz/J7l8/HfRfP0X5Xz8+/l8/5wJgPz4HYD9zC2A/XQ9gPwUTYD9cFmA/"
        "rBlgP8ocYD+9H2A/niJgPxslYD+UJ2A//ylgPw4sYD9ZLmA/LzBgPygyYD/8M2A/"
        "t2NWP5y4WD8pFFo/utFaP3dIWz/Dn1s/kuVbP/8fXD+pUlw/6n9cPyapXD8Hz1w/"
        "FfJcP0YSXT+oL10/gEpdP8diXT+jeF0/SIxdP+qdXT+NrV0/f7tdPwfIXT8w010/"
        "Dt1dPwnmXT8u7l0/TPVdP+b7XT/gAV4/VAdePy0MXj/sEF4/KBVePycZXj/JHF4/"
        "QyBeP3YjXj94Jl4/TylePxQsXj+aLl4//zBePzwzXj9VNV4/cDdeP1Y5Xj8/O14/"
        "+DxeP5s+Xj8WQF4/sUFePwhDXj9vRF4/nkVeP8tGXj//R14/LUlePymnST+4DVM/"
        "VjhWP7VFVz/tzVc/qm5YP85TWT9ZeVo//stbP2g3XT81q14/rRtgP0GBYT8V12I/"
        "5RpkP4dLZT/4aGY/snNnP5psaD+mVGk/zyxqP3n2aj9csms/vGFsP5sFbT+Onm0/"
        "sS1uP6ezbj8GMW8/laZvP9AUcD8efHA/GN1wPxg4cT+bjXE/wt1xPxcpcj/lb3I/"
        "U7JyP7Lwcj84K3M/LWJzP8uVcz92xnM/7/NzP8cedD/rRnQ/tmx0PziQdD+csXQ/"
        "/dB0P57udD+BCnU/oCR1P2w9dT+3VHU/wWp1P35/dT8="),
    "cld/extice2": ("f4", (43, 14),
        "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"
        "qu5VPG6IUDw="),
    "cld/ssaice2": ("f4", (43, 14),
        "Zr4pP2ImJT+DJCI/3OsfP+UqHj/7uBw/BX8bP6BuGj9Gfhk/UKcYP/LkFz+aMxc/"
        "hZAWP5T5FT8RbRU/nOkUPxNuFD+K+RM/NYsTP24iEz+ivhI/Vl8SPx8EEj+grBE/"
        "iFgRP44HET9yuRA//G0QP/skED9A3g8/pJkPPwJXDz84Fg8/K9cOP7yZDj/WXQ4/"
        "XyMOP0PqDT9ysg0/1XsNP2BGDT8DEg0/r94MP4rbRD+nZD0/Dng4P7XhND/sGDI/"
        "m9gvPwr5LT9cYSw//AArP1fMKT8Duyg/w8YnP8PqJj9FIyY/Tm0lP3PGJD+6LCQ/"
        "hJ4jP28aIz9WnyI/PiwiP07AIT/OWiE/GvsgP6SgID/ySiA/l/kfPzCsHz9oYh8/"
        "8RsfP4bYHj/qlx4/5VkeP0IeHj/W5B0/da0dP/t3HT9CRB0/LxIdP5/hHD97shw/"
        "qoQcPxNYHD+84H4/FEJ7P3a7eD8Kx3Y/AC11P43Qcz/foHI/OZNxP1mgcD8bw28/"
        "tfduP0k7bj+hi20/+OZsP+RLbD8/uWs/FS5rP5Wpaj8VK2o//rFpP9Q9aT8mzmg/"
        "mGJoP8/6Zz+Hlmc/ezVnP3DXZj8zfGY/lCNmP2jNZT+JeWU/0ydlPyjYZD9pimQ/"
        "fT5kP0z0Yz+8q2M/vWRjPzgfYz8d22I/XJhiP+NWYj+mFmI/0Ol4P2j5cT8XQG0/"
        "SKlpP7bDZj+/VWQ//j1iP19nYD+yw14//khdP/DvWz8Gs1o/AI5ZP4J9WD/bflc/"
        "2I9WP6quVT/N2VQ/+Q9UPxdQUz85mVI/kOpRP2hDUT8go1A/MglQPx51Tz935k4/"
        "3FxOP/XXTT9zV00/DNtMP4JiTD+X7Us/F3xLP9ANSz+Toko/NzpKP5jUST+QcUk/"
        "/xBJP8eySD/IVkg/7fxHP9ZKfj+BcHw/ES17P/A2ej//b3k/58h4P784eD/xuXc/"
        "tEh3P13idj/8hHY/GC92P5HfdT+BlXU/MFB1PwoPdT+R0XQ/Xpd0PxxgdD+BK3Q/"
        "SvlzP0TJcz87m3M/CW9zP4VEcz+RG3M/EPRyP+XNcj/8qHI/PoVyP5hicj/5QHI/"
        "USByP5IAcj+v4XE/nMNxP02mcT+4iXE/1G1xP5lScT//N3E//B1xP4sEcT/MdX0/"
        "jJ57P1hZej/AX3k/qZR4PyTpdz+BVHc/O9F2P5pbdj/78HU/bo91P301dT8H4nQ/"
        "J5R0PyNLdD9lBnQ/cMVzP96Hcz9TTXM/hhVzPzbgcj8srXI/NHxyPyZNcj/aH3I/"
        "L/RxPwjKcT9KoXE/23lxP6ZTcT+ZLnE/oApxP63ncD+vxXA/mKRwP1+EcD/0ZHA/"
        "T0ZwP2UocD8wC3A/ou5vP7nSbz9ot28/cql/P0Offz+Pln8/uY5/P3SHfz+cgH8/"
        "F3p/P9dzfz/MbX8/8md/P0Bifz+xXH8/QFd/P+xRfz+yTH8/jUd/P31Cfz+BPX8/"
        "lTh/P7kzfz/vLn8/MCp/P4Elfz/bIH8/Qxx/P7UXfz8zE38/uw5/P0sKfz/jBX8/"
        "hgF/Py/9fj/h+H4/mvR+P1rwfj8i7H4/8Od+P8Tjfj+f334/ftt+P2bXfj9Q034/"
        "Qs9+P4fyfz+G8X8/sfB/P+3vfz8y738/e+5/P8btfz8P7X8/Vex/P5jrfz/X6n8/"
        "Eep/P0fpfz936H8/oud/P8jmfz/p5X8/BeV/Pxrkfz8o438/M+J/Pzbhfz814H8/"
        "Lt9/PyDefz8N3X8/9dt/P9bafz+w2X8/hdh/P1bXfz8f1n8/5NR/P6LTfz9b0n8/"
        "DtF/P7zPfz9izn8/Bc1/P6DLfz83yn8/yMh/P1PHfz+q/38/p/9/P6D/fz+a/38/"
        "kf9/P4b/fz96/38/bv9/P1//fz9Q/38/P/9/Py3/fz8a/38/Bv9/P/D+fz/Z/n8/"
        "wf5/P6j+fz+P/n8/dP5/P1j+fz85/n8/G/5/P/v9fz/b/X8/uv1/P5f9fz9z/X8/"
        "Tv1/Pyj9fz8B/X8/2fx/P7H8fz+H/H8/W/x/PzD8fz8E/H8/1ft/P6b7fz93+38/"
        "Rvt/PxT7fz/i+n8/+/9/P/v/fz/7/38/+f9/P/n/fz/4/38/9v9/P/T/fz/0/38/"
        "8/9/P/H/fz/u/38/7P9/P+r/fz/p/38/5f9/P+P/fz/g/38/3v9/P9v/fz/Y/38/"
        "1P9/P9H/fz/P/38/zP9/P8f/fz/E/38/wP9/P73/fz+4/38/tf9/P7H/fz+s/38/"
        "p/9/P6T/fz+f/38/mv9/P5X/fz+R/38/jP9/P4f/fz+C/38/e/9/P/v/fz/5/38/"
        "+f9/P/j/fz/2/38/9P9/P/P/fz/x/38/7/9/P+7/fz/q/38/6f9/P+X/fz/i/38/"
        "3v9/P9v/fz/Y/38/1P9/P9H/fz/O/38/yf9/P8X/fz/A/38/u/9/P7b/fz+z/38/"
        "rv9/P6f/fz+i/38/nf9/P5j/fz+R/38/jP9/P4b/fz9//38/ev9/P3P/fz9s/38/"
        "Zv9/P1//fz9X/38/UP9/P0n/fz/q/38/6v9/P+f/fz/l/38/4v9/P97/fz/b/38/"
        "2P9/P9P/fz/O/38/yf9/P8T/fz+9/38/tv9/P6//fz+p/38/ov9/P5r/fz+R/38/"
        "if9/P4D/fz92/38/bv9/P2T/fz9a/38/Tv9/P0T/fz84/38/Lf9/PyH/fz8V/38/"
        "Cf9/P/z+fz/v/n8/4f5/P9T+fz/G/n8/t/5/P6r+fz+b/n8/jP5/P3v+fz9s/n8/"
        "7v9/P+7/fz/q/38/6f9/P+X/fz/i/38/3v9/P9v/fz/W/38/0f9/P8z/fz/H/38/"
        "wP9/P7r/fz+z/38/rP9/P6T/fz+d/38/lf9/P4z/fz+C/38/ev9/P3D/fz9m/38/"
        "XP9/P1L/fz9G/38/PP9/PzD/fz8k/38/F/9/Pwv/fz/+/n8/8P5/P+P+fz/V/n8/"
        "yP5/P7n+fz+q/n8/m/5/P4z+fz98/n8/bP5/P2NHND8xSys/fXUlP1UxIT//2x0/"
        "BCQbP/HaGD/d4xY/nisVP9ykEz/8RRI/0gcRP+rkDz8B2Q4/t+ANP1P5DD+eIAw/"
        "wVQLPzSUCj+v3Qk/GDAJP3+KCD8U7Ac/JVQHPxLCBj9UNQY/cK0FP/wpBT+XqgQ/"
        "7C4EP662Az+ZQQM/bs8CP/RfAj/58gE/TIgBP8YfAT8+uQA/kVQAPzvj/z6JIP8+"
        "2GD+Pu6j/T4="),
    "cld/asyice2": ("f4", (43, 14),
        "M29LPxzSWj8bb2E/7x1lP4d3Zz8lGWk/LkxqP4Y3az+88Ws/xohsP8kFbT/4bm0/"
        "ushtPzsWbj/YWW4/W5VuPyfKbj9S+W4/tyNvPwxKbz/hbG8/q4xvP9Kpbz+jxG8/"
        "ad1vP170bz+1CXA/mR1wPy8wcD+aQXA/9FFwP1lhcD/fb3A/ln1wP5OKcD/klnA/"
        "l6JwP7mtcD9UuHA/csJwPxvMcD9a1XA/Nd5wPy1maD/saWs/2T1tP2+Ebj+Eem8/"
        "GT1wPy7ccD94YXE/R9NxP+k1cj9yjHI/JtlyP7gdcz96W3M/dpNzP4DGcz9E9XM/"
        "USB0PxlIdD8CbXQ/Wo90P2+vdD92zXQ/qel0PzMEdT87HXU/5zR1P1NLdT+cYHU/"
        "2XR1Px+IdT+BmnU/E6x1P+S8dT8AzXU/dtx1P1HrdT+b+XU/Xgd2P6EUdj9wIXY/"
        "zy12P8g5dj9bC1U/pFVaP6FkXT9xd18/8v9gP5gyYj92K2M/KPtjPz6sZD/QRWU/"
        "1MxlP+REZj+usGY/PhJnPy1rZz++vGc/9gdoP6dNaD+Cjmg/F8toP+ADaT9IOWk/"
        "pGtpP0KbaT9lyGk/RPNpPxUcaj8DQ2o/MmhqP8qLaj/mrWo/ps5qPx7uaj9pDGs/"
        "mClrP79Faz/uYGs/NntrP6SUaz9ErWs/I8VrP0rcaz/F8ms/sMhPP8dMWT8YEF4/"
        "AvpgP/n2Yj8ea2Q/johlP0xrZj9TJGc/jb5nP1RBaD/QsWg/vBNpP9hpaT8/tmk/"
        "jPppPwc4aj+xb2o/XqJqP7jQaj9M+2o/kSJrP+pGaz+qaGs/GohrP3alaz/0wGs/"
        "wtprPwjzaz/pCWw/hh9sP/gzbD9YR2w/vVlsPztrbD/ge2w/wItsP+WabD9fqWw/"
        "OrdsP3/EbD840Ww/bN1sP48BTj8CHVc/h2ZbP5bpXT/wkV8/1r9gPxiiYT9BUmI/"
        "et9iP1dTYz8wtGM/aQZkPyFNZD+jimQ/pcBkP3nwZD8hG2U/bkFlPwVkZT9sg2U/"
        "EKBlP026ZT9v0mU/tOhlP1L9ZT96EGY/USJmP/kyZj+SQmY/NVFmP/ZeZj/ta2Y/"
        "J3hmP7mDZj+ujmY/EZlmP/CiZj9UrGY/RLVmP8y9Zj/yxWY/u81mPy/VZj+yYlE/"
        "SRtZP8e4XD/b1F4/EThgPxE0YT9v8GE/xIJiP9H3Yj+qV2M/oKdjP2TrYz+RJWQ/"
        "E1hkP1uEZD+Aq2Q/XM5kP5vtZD/KCWU/UiNlP5Q6ZT/aT2U/YmNlP2F1ZT8FhmU/"
        "cpVlP8yjZT8ssWU/rb1lP2TJZT9j1GU/ut5lP3foZT+r8WU/XfplP5gCZj9oCmY/"
        "0xFmP+EYZj+XH2Y//iVmPxksZj/vMWY/hyVRP9hyWD9G01s/scddP/AOXz8X9l8/"
        "TaJgP7YnYT8ykmE/OOlhP7cxYj8Sb2I/sqNiP1fRYj9V+WI/qBxjPxo8Yz9IWGM/"
        "rXFjP6+IYz+lnWM/zrBjP2fCYz+c0mM/m+FjP4HvYz9v/GM/fQhkP8ITZD9RHmQ/"
        "OihkP48xZD9ZOmQ/pUJkP39KZD/vUWQ//lhkP7BfZD8RZmQ/JGxkP+9xZD92d2Q/"
        "vnxkP0C5Vj83ZFs/M6ldP9EFXz9O8F8/05lgP6YaYT8ugGE/e9JhP6oWYj8xUGI/"
        "b4FiPx6sYj+D0WI/kvJiPwkQYz94KmM/V0JjPwNYYz/Ka2M/631jP5qOYz8DnmM/"
        "SqxjP5O5Yz/0xWM/h9FjP2LcYz+S5mM/KfBjPzb5Yz/CAWQ/2glkP4YRZD/QGGQ/"
        "vx9kP1gmZD+lLGQ/pzJkP2Y4ZD/mPWQ/KUNkPzdIZD8kK1o/ZmRdP+EBXz9a/18/"
        "t6xgP8grYT94jWE/QdthP98aYj8CUGI/K31iPxWkYj8IxmI/8eNiP4b+Yj9VFmM/"
        "zStjP0Y/Yz8HUWM/SWFjP0FwYz8TfmM/44pjP8+WYz/soWM/UaxjPxC2Yz85v2M/"
        "28djPwDQYz+112M/Ad9jP+7lYz+C7GM/xvJjP8D4Yz90/mM/5gNkPxwJZD8aDmQ/"
        "4xJkP30XZD/mG2Q/G5pbPyBJXj+FoF8/nHJgPxECYT8Qa2E/r7thP877YT80MGI/"
        "7VtiPw6BYj8HoWI/57xiP3LVYj8+62I/wP5iP1cQYz9IIGM/zy5jPx08Yz9ZSGM/"
        "plNjPx1eYz/YZ2M/7HBjP2d5Yz9cgWM/1IhjP92PYz+AlmM/x5xjP7miYz9cqGM/"
        "uK1jP9GyYz+tt2M/T7xjP73AYz/5xGM/B8ljP+zMYz+m0GM/PNRjP9XQXD/2El8/"
        "ASZgP0fIYD/oM2E/yoBhP5e6YT+452E/+wtiP8kpYj/AQmI/+ldiP0FqYj8semI/"
        "KohiP5KUYj+kn2I/lqliP5SyYj+/umI/NMJiPwjJYj9Rz2I/HtViP37aYj9632I/"
        "IeRiP3roYj+K7GI/WvBiP/DzYj9S92I/gfpiP4X9Yj9gAGM/FQNjP6cFYz8ZCGM/"
        "bApjP6UMYz/DDmM/yRBjP7gSYz8C11s/eFRePw9xXz/SEmA/gntgPwDFYD+R+2A/"
        "wiVhP2tHYT/wYmE/4XlhP1KNYT8GnmE/iqxhP0m5YT+SxGE/o85hP6/XYT/b32E/"
        "SOdhPw7uYT9H9GE///lhP0n/YT8wBGI/vghiP/8MYj/5EGI/shRiPzIYYj9+G2I/"
        "mx5iP4whYj9UJGI/+CZiP3opYj/bK2I/IS5iP00wYj9fMmI/WjRiPz82Yj8POGI/"
        "FypbP/YrXj/mbF8/ghlgPxWEYD/gy2A/Qv9gP64lYT9oQ2E/B1thPzduYT8TfmE/"
        "Y4thP7SWYT9voGE/4qhhP0ewYT/OtmE/l7xhP8TBYT9sxmE/ncphP2zOYT/k0WE/"
        "ENVhP/fXYT+l2mE/HN1hP2XfYT+F4WE/fuNhP1XlYT8N52E/quhhPy3qYT+Y62E/"
        "7uxhPzLuYT9h72E/gvBhP5PxYT+W8mE/i/NhP8jDVD/4al4/1thjPxtxZz8zC2o/"
        "YQlsP8ufbT8Q7W4/hARwPwrzcD+awXE/p3ZyP/sWcz82pnM/Jyd0PwacdD+UBnU/"
        "RGh1Pz/CdT+BFXY/2WJ2P/Wqdj9t7nY/vy13P1ppdz+doXc/29Z3P1wJeD9iOXg/"
        "Imd4P9KSeD+dvHg/qeR4Px0LeT8YMHk/tlN5PxN2eT9Gl3k/Zbd5P4LWeT+09Hk/"
        "BxJ6P4suej8="),
    "cld/extice3": ("f4", (46, 14),
        "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"),
    "cld/ssaice3": ("f4", (46, 14),
        "JcksPxg9Kj/PFSg/ZCQmP2paJD+ksSI/rSYhP0O3Hz+zYR4/liQdP6X+Gz+17ho/"
        "qfMZP2oMGT/nNxg/FHUXP+fCFj9YIBY/XowVP/QFFT8UjBQ/uB0UP9+5Ez9/XxM/"
        "mA0TPyTDEj8ifxI/jEASP18GEj+YzxE/OJsRPzZoET+UNRE/TwIRP2TNED/RlRA/"
        "lFoQP6waED8W1Q8/04gPP+E0Dz882A4/53EOP98ADj8lhA0/tfoMP3dKQz8KuUE/"
        "dzdAPz7FPj/dYT0/0Aw8P5fFOj+3izk/q144P/k9Nz8jKTY/rB81PxghND/vLDM/"
        "tkIyP/FhMT8rijA/7bovP77zLj8nNC4/t3stP/bJLD90Hiw/ungrP1nYKj/fPCo/"
        "26UpP90SKT95gyg/PfcnP71tJz+N5iY/QWEmP2zdJT+nWiU/htgkP6BWJD+O1CM/"
        "6VEjP0jOIj9HSSI/f8IhP485IT8PriA/nx8gP9uNHz/kDn4/3v98Pzf2ez/P8Xo/"
        "gvJ5Pyr4eD+nAng/0xF3P4wldj+wPXU/GFp0P6R6cz8yn3I/m8dxP8HzcD99I3A/"
        "sFZvPzSNbj/pxm0/rANtP1pDbD/RhWs/7cpqP48Saj+TXGk/16hoPzr3Zz+YR2c/"
        "0plmP8PtZT9MQ2U/SZpkP5vyYz8gTGM/tKZiPzkCYj+KXmE/irtgPxUZYD8Ld18/"
        "S9VeP7QzXj8kkl0/fPBcP5tOXD9frFs/UoN2P86jcz8x4HA/pjduP1upaz95NGk/"
        "L9hmP6eTZD8RZmI/l05gP2xMXj+6Xlw/s4RaP4O9WD9aCFc/a2RVP+XQUz/1TFI/"
        "09dQP6twTz+vFk4/FMlMPwyHSz/HT0o/fSJJP13+Rz+e4kY/dM5FPxPBRD+xuUM/"
        "grdCP725QT+Xv0A/SMg/PwXTPj8H3z0/hus8P7b3Oz/SAjs/FAw6P7ISOT/lFTg/"
        "6RQ3P/UONj9EAzU/EvEzP6mFfj/5rH0/b9h8P+4HfD9eO3s/nnJ6P5WteT8p7Hg/"
        "OS54P61zdz9nvHY/TAh2P0BXdT8mqXQ/5P1zP1pVcz9xr3I/CAxyPwZrcT9NzHA/"
        "wy9wP0qVbz/H/G4/HWZuPy7RbT/iPW0/GaxsP7kbbD+jjGs/v/5qP+xxaj8Q5mk/"
        "D1tpP8vQaD8pR2g/C75nP1g1Zz/wrGY/tyRmP5KcZT9jFGU/DYxkP3YDZD9/emM/"
        "DvFiPwJnYj+OT34/QWx9P8mOfD/+tns/teR6P8gXej8NUHk/W414P4vPdz9xFnc/"
        "6WF2P8exdT/lBXU/GV50Pz26cz8lGnM/rH1yP6rkcT/2TnE/abxwP9kscD8foG8/"
        "FRZvP5GObj9tCW4/gYZtP6QFbT+xhmw/fQlsP+SNaz++E2s/4ppqPysjaj9vrGk/"
        "iTZpP1PBaD+jTGg/VdhnP0FkZz8+8GY/KHxmP9gHZj8nk2U/7R1lPwWoZD9IMWQ/"
        "I95/P+/Lfz+7uX8/h6d/P1WVfz8ig38/83B/P8Nefz+VTH8/Zzp/Pzwofz8SFn8/"
        "6gN/P8Pxfj+f334/fc1+P1y7fj89qX4/Ipd+PwiFfj/ycn4/3mB+P81Ofj+/PH4/"
        "sip+P6sYfj+lBn4/o/R9P6XifT+q0H0/s759P76sfT/Omn0/44h9P/p2fT8WZX0/"
        "N1N9P1lBfT+DL30/rx19P+ELfT8Y+nw/Uuh8P5HWfD/VxHw/HrN8P7L7fz/n+H8/"
        "HvZ/P1Xzfz+M8H8/xe1/P//qfz856H8/dOV/P7Hifz/t338/Ld1/P2rafz+q138/"
        "69R/PyzSfz9uz38/scx/P/bJfz86x38/gMR/P8bBfz8Mv38/VLx/P525fz/ntn8/"
        "MLR/P3uxfz/Grn8/E6x/P2Cpfz+tpn8/+6N/P0uhfz+bnn8/7Jt/PzyZfz+Nln8/"
        "4ZN/PzSRfz+Ijn8/3Yt/PzKJfz+Hhn8/3oN/PzWBfz/b/38/xf9/P6//fz+a/38/"
        "hP9/P27/fz9Y/38/Qv9/Pyv/fz8V/38///5/P+n+fz/U/n8/vv5/P6j+fz+S/n8/"
        "fP5/P2f+fz9R/n8/O/5/PyX+fz8P/n8/+v1/P+T9fz/O/X8/uP1/P6L9fz+N/X8/"
        "d/1/P2H9fz9L/X8/Nf1/Px/9fz8K/X8/9Px/P978fz/I/H8/svx/P538fz+H/H8/"
        "cfx/P1v8fz9F/H8/MPx/Pxr8fz8E/H8/9v9/P/T/fz/x/38/7/9/P+7/fz/q/38/"
        "6f9/P+X/fz/j/38/4v9/P97/fz/d/38/2f9/P9j/fz/W/38/0/9/P9H/fz/O/38/"
        "zP9/P8n/fz/H/38/xf9/P8L/fz/A/38/vf9/P7v/fz+4/38/tv9/P7P/fz+x/38/"
        "r/9/P6z/fz+q/38/p/9/P6X/fz+i/38/oP9/P53/fz+b/38/mv9/P5b/fz+V/38/"
        "kf9/P5D/fz+M/38/i/9/P/v/fz/4/38/8/9/P+//fz/q/38/5/9/P+P/fz/e/38/"
        "2/9/P9j/fz/T/38/z/9/P8z/fz/J/38/xf9/P8L/fz+9/38/uv9/P7b/fz+z/38/"
        "r/9/P6z/fz+p/38/pf9/P6L/fz+f/38/mv9/P5b/fz+T/38/kP9/P4z/fz+J/38/"
        "hv9/P4L/fz9//38/ev9/P3b/fz9z/38/cP9/P2v/fz9n/38/ZP9/P2H/fz9c/38/"
        "WP9/P1P/fz/x/38/5/9/P97/fz/U/38/yv9/P8L/fz+4/38/rv9/P6X/fz+b/38/"
        "kf9/P4n/fz9//38/dv9/P2z/fz9k/38/Wv9/P1L/fz9J/38/P/9/Pzf/fz8t/38/"
        "JP9/Pxr/fz8S/38/Cf9/P//+fz/3/n8/7f5/P+T+fz/c/n8/0v5/P8r+fz/A/n8/"
        "t/5/P63+fz+l/n8/m/5/P5L+fz+I/n8/fv5/P3b+fz9s/n8/Y/5/P1n+fz9P/n8/"
        "1v9/P7//fz+l/38/jP9/P3X/fz9d/38/RP9/Py3/fz8T/38//P5/P+T+fz/L/n8/"
        "tP5/P5z+fz+F/n8/bP5/P1T+fz89/n8/Jf5/Pwz+fz/1/X8/3f1/P8b9fz+u/X8/"
        "lf1/P339fz9m/X8/Tv1/PzX9fz8e/X8/Bv1/P+/8fz/W/H8/vvx/P6f8fz+O/H8/"
        "dvx/P1/8fz9F/H8/Lvx/Pxb8fz/9+38/5vt/P837fz+1+38/nPt/P77B5z63Xgc/"
        "K4IKP8XFCz95aww/r8UMP2/zDD9DAw0/EP0MP6jlDD9IwAw/Ro8MP3VUDD9WEQw/"
        "M8cLPzd3Cz92Igs/9skKP69uCj+XEQo/nrMJP7ZVCT/N+Ag/050IP7tFCD948Qc/"
        "AqIHP1JYBz9oFQc/RdoGP/CnBj90fwY/4mEGP09QBj/VSwY/lVUGP7ZuBj9hmAY/"
        "zdMGPy4iBz/HhAc/3vwHP76LCD++Mgk/OvMJP5POCj8="),
    "cld/asyice3": ("f4", (46, 14),
        "9IVVP7vwVz80DFo/8/RbP7a0XT9kUF8/JMtgP2YnYj8yZ2M/U4xkP3eYZT8ojWY/"
        "4mtnPwk2aD/57Gg/+ZFpP0gmaj8cq2o/niFrP+2Kaz8j6Gs/UDpsP32CbD+vwWw/"
        "4/hsPxApbT8uU20/LXhtP/yYbT+Etm0/rNFtP13rbT94BG4/4h1uP3c4bj8YVW4/"
        "oXRuP+qXbj/Lv24/Ge1uP6Ygbz9AW28/sp1vP77obz8jPXA/nJtwP99wXz8hs2A/"
        "W+NhP0cCYz+dEGQ/EA9lP1H+ZT8K32Y/5bFnP4h3aD+WMGk/rt1pP2x/aj9nFms/"
        "PKNrP3gmbD+zoGw/ehJtP1x8bT/i3m0/mjpuPwmQbj+4324/KypvP+hvbz9xsW8/"
        "Su9vP/EpcD/nYXA/r5dwP8TLcD+m/nA/0TBxP8FicT/ylHE/4MdxPwT8cT/VMXI/"
        "z2lyP2Skcj8M4nI/OSNzP11ocz/qsXM/SgB0P+tTdD99LUo/oHxLP/a/TD/L900/"
        "bCRPPyNGUD8+XVE/CmpSP85sUz/bZVQ/elVVP/U7Vj+YGVc/rO5XP3y7WD9TgFk/"
        "dz1aPzjzWj/aoVs/qklcP+3qXD/whV0/+xpeP1eqXj9KNF8/H7lfPx05YD+PtGA/"
        "uSthP+WeYT9aDmI/YnpiP0LjYj9CSWM/qqxjP8MNZD/RbGQ/HcpkP+wlZT+IgGU/"
        "NtplPzszZj/gi2Y/a+RmPyE9Zz9Jlmc/wQBIP0DtST90xEs/AodNP441Tz+60FA/"
        "KVlSP3rPUz9PNFU/QohWP/DLVz/1/1g/6yRaP2k7Wz8IRFw/XT9dP/4tXj+AEF8/"
        "c+dfP22zYD/+dGE/tyxiPynbYj/jgGM/dB5kP2q0ZD9UQ2U/wMtlPzlOZj9Qy2Y/"
        "jUNnP363Zz+vJ2g/q5RoPwD/aD82Z2k/2M1pP3Qzaj+RmGo/uf1qP3Zjaz9Syms/"
        "0zJsP4KdbD/nCm0/h3ttPzmaQz+E5EQ/eiRGP1daRz9Whkg/s6hJP6zBSj980Us/"
        "XthMP47WTT9FzE4/v7lPPzefUD/kfFE/AlNSP8shUz946VM/QqpUP2FkVT8PGFY/"
        "hcVWP/xsVz+rDlg/yapYP5JBWT8901k/AWBaPxfoWj+1a1s/FetbP2tmXD/13Vw/"
        "5lFdP3bCXT/cL14/U5pePw4CXz9IZ18/NcpfPxArYD8NimA/Y+dgP0xDYT/8nWE/"
        "rPdhP5JQYj9joEI/WOFDP0IYRT9iRUY/9GhHPzeDSD9klEk/t5xKP2ucSz+4k0w/"
        "14JNP/9pTj9rSU8/TSFQP97xUD9Ru1E/3n1SP7U5Uz8N71M/GZ5UPwtHVT8U6lU/"
        "aYdWPzgfVz+0sVc/DT9YP3XHWD8aS1k/LcpZP9xEWj9Wu1o/yy1bP2mcWz9fB1w/"
        "225cPwjTXD8WNF0/MJJdP4XtXT8/Rl4/jZxeP5nwXj+QQl8/m5JfP+fgXz+fLWA/"
        "68NBPzS9Qj8xsEM//JxEP6mDRT9RZEY/Cz9HP+4TSD8S40g/jqxJP3dwSj/qLks/"
        "+edLP7+bTD9SSk0/yfNNPzqYTj/AN08/cNJPP2FoUD+s+VA/aIZRP6wOUj+PklI/"
        "KBJTP5GNUz/fBFQ/K3hUP4rnVD8XU1U/5rpVPxAfVj+tf1Y/1dxWP5w2Vz8cjVc/"
        "beBXP6YwWD/efVg/LMhYP6kPWT9sVFk/ipZZPx7WWT8+E1o/Ak5aP/TrQT+pvEI/"
        "8olDP8pTRD8sGkU/E91FP36cRj9mWEc/xxBIP57FSD/mdkk/nCRKP7jOSj85dUs/"
        "GxhMP1i3TD/tUk0/1+pNPw5/Tj+RD08/XJxPP2olUD+3qlA/PyxRP/6pUT/vI1I/"
        "DppSP1gMUz/LelM/XuVTPxBMVD/drlQ/wQ1VP7doVT++v1U/zRJWP+RhVj//rFY/"
        "GPRWPyw3Vz84dlc/OLFXPyfoVz8AG1g/w0lYP2l0WD8Uw0E/6IFCP6M+Qz8z+UM/"
        "hLFEP4lnRT8vG0Y/aMxGPyB7Rz9IJ0g/0NBIP6V3ST+3G0o/97xKP1RbSz+99ks/"
        "IY9MP3EkTT+btk0/jkVOPz3RTj+SWU8/g95PP/pfUD/r3VA/Q1hRP/TOUT/rQVI/"
        "G7FSP3AcUz/dg1M/UOdTP7pGVD8LolQ/M/lUPyFMVT/GmlU/EeVVP/IqVj9bbFY/"
        "OqlWP4HhVj8dFVc/A0RXPx5uVz9hk1c/nD5BP3/uQT94nUI/bEtDPz/4Qz/Vo0Q/"
        "Ek5FP9j2RT8OnkY/lkNHP1PnRz8riUg/ASlJP7rGST84Yko/YvtKPxmSSz9DJkw/"
        "wrdMP3tGTT9S0k0/LFtOP+3gTj94Y08/sOJPP3peUD+51lA/VUtRPy28UT8oKVI/"
        "KZJSPxT3Uj/NV1M/OrRTPzwMVD+5X1Q/la5UP7L4VD/2PVU/RH5VP4K5VT+S71U/"
        "WSBWP7pLVj+bcVY/35FWP53KPz9keUA/fydBP87UQT8zgUI/kCxDP8XWQz+2f0Q/"
        "QidFP03NRT+4cUY/YxRHPy+1Rz8BVEg/t/BIPzWLST9bI0o/DrlKPypMSz+U3Es/"
        "LWpMP9b0TD9zfE0/4QBOPwaCTj/B/04/9HlPP4LwTz9JY1A/MNJQPxQ9UT/Yo1E/"
        "XQZSP4dkUj81vlI/SRNTP6NjUz8pr1M/ufVTPzc3VD+Cc1Q/fKpUPwjcVD8GCFU/"
        "Wi5VP+ROVT+LXT0/YQ4+P5S+Pj8Fbj8/lhxAPyfKQD+ZdkE/ziFCP6bLQj8BdEM/"
        "wBpEP8W/RD/zYkU/KARGP0WjRj8rQEc/vNpHP9hySD9hCEk/NptJPzorSj9PuEo/"
        "UkJLPyXJSz+sTEw/xMxMP1JJTT8zwk0/STdOP3ioTj+cFU8/mX5PP1HjTz+gQ1A/"
        "bJ9QP5b2UD/6SFE/fZZRP//eUT9hIlI/hGBSP0iZUj+PzFI/OfpSPyciUz88RFM/"
        "s/I4P56kOT/fVTo/VgY7P+i1Oz90ZDw/3hE9Pwa+PT/RaD4/HhI/P9G5Pz/KX0A/"
        "7ANBPxumQT80RkI/HuRCP7h/Qz/mGEQ/iK9EP4BDRT+y1EU//mJGP0fuRj9vdkc/"
        "V/tHP+N8SD/y+kg/aXVJPyjsST8RX0o/B85KP+o4Sz+fn0s/BgJMPwFgTD9yuUw/"
        "PQ5NP0JeTT9iqU0/f+9NP38wTj8/bE4/pKJOP5DTTj/k/k4/gCRPP4iPYj9fimY/"
        "LyxoP5+EaT/guWo/dtVrPxDbbD+TzG0/NqtuP+p3bz+AM3A/ud5wP156cT8qB3I/"
        "54VyP1/3cj9fXHM/urVzP0kEdD/nSHQ/dIR0P9G3dD/j43Q/lgl1P88pdT9+RXU/"
        "kF11P/RydT+ahnU/dpl1P32sdT+gwHU/2tZ1PyLwdT9zDXY/yi92PydYdj+Qh3Y/"
        "CL92P57/dj9jSnc/bKB3P9YCeD/Icng/avF4P/B/eT8="),
    "cld/fdlice3": ("f4", (46, 14),
        "yiFLPdnoPz0QSzU9Y0MrPcHMIT0Z4hg9Wn4QPXmcCD1lNwE9HJT0PMWe5zyrhNs8"
        "sjvQPLO5xTyY9Ls8O+KyPHp4qjxEraI8aXabPNbJlDxtnY48BOeIPIicgzyqZ308"
        "l0V0PJW+azxrvmM84zBcPK8BVTyZHE48Z21HPNrfQDy4Xzo8tdgzPJo2LTwvZSY8"
        "LFAfPF3jFzx/ChA8VrEHPESH/TtPWuo7TrPVO8dpvzs+Vac7Nk2NO6O3Tz0Tz0w9"
        "ZhdKPYqORz1vMkU9BAFDPTP4QD3tFT89IFg9PbS8Oz2gQTo9y+Q4PSakNz2gfTY9"
        "I281PZ92ND0AkjM9OL8yPTH8MT3dRjE9JJ0wPfj8Lz1HZC89/NAuPQZBLj1Vsi09"
        "1iItPXSQLD0i+Ss9x1orPVizKj3AACo96UApPcdxKD1HkSc9UJ0mPdmTJT3MciQ9"
        "EzgjPaThIT1mbSA9StkePTwjHT0sSRs9B0kZPbsgFz2QsNk98SjaPaCw2j2TRts9"
        "runbPQCZ3D1wU909/hfePZ7l3j1Eu98945fgPYl64T0dYuI9kk3jPdw75D0JLOU9"
        "8hzmPaYN5z0X/ec9LOroPfTT6T1Uueo9TpnrPdVy7D3dRO09WQ7uPVfO7j2xg+89"
        "Zy3wPYjK8D3fWfE9iNrxPXdL8j2Rq/I9yvnyPTE18z2sXPM9O2/zPdNr8z1mUfM9"
        "6R7zPVzT8j2zbfI98OzxPelP8T29lfA9nWfcPeUm2j1yANg99PLVPQD90z1hHdI9"
        "vFLQPcybzj0q98w9pmPLPdzfyT2Basg9RQLHPdqlxT3wU8Q9NgvDPV/KwT0YkMA9"
        "FFu/PQUqvj2Y+7w9fs67PWmhuj0Jc7k9DkK4PSgNtz0I07U9YJK0Pd1Jsz00+LE9"
        "EZywPSc0rz0mv609vTusPZ+oqj16BKk9AU6nPeKDpT3PpKM9eK+hPY2inz2/fJ09"
        "vjybPTvhmD3naJY9cNKTPZAs5T0h7uU9Ab/mPTKe5z2Yiug9RIPpPQ2H6j30lOs9"
        "7avsPd/K7T3X8O49rhzwPVhN8T3jgfI9KLnzPSjy9D3WK/Y9GWX3PfKc+D1U0vk9"
        "JQT7PWcx/D0MWf09+3n+PUOT/z3XUQA+LdUAPg9TAT79ygE+azwCPtmmAj7BCQM+"
        "l2QDPtm2Az4DAAQ+iD8EPud0BD6VnwQ+EL8EPs7SBD5N2gQ+CNUEPnLCBD4LogQ+"
        "R3MEPqY1BD50Geg928/oPfuV6T3Kauo9O03rPTQ87D2nNu09bzvuPZpJ7z3/X/A9"
        "hn3xPTCh8j3hyfM9gvb0PQQm9j1PV/c9ZIn4PRq7+T106/o9SRn8PY1D/T00af49"
        "JYn/PSlRAD7R2QA+BF4BPjzdAT7sVgI+h8oCPoY3Az5enQM+j/sDPopRBD6+ngQ+"
        "sOIEPs4cBT6LTAU+ZnEFPtSKBT5HmAU+OJkFPiONBT55cwU+rUsFPjsVBT6izwQ+"
        "2pHqPdQv7D2W3e09+ZnvPQtk8T2YOvM9nxz1PQkJ9z26/vg9pvz6PaYB/T3IDP89"
        "bI4APmWYAT7IowI+ArADPpm8BD71yAU+lNQGPuTeBz5e5wg+fO0JPrLwCj5x8As+"
        "NOwMPm/jDT6T1Q4+HMIPPoKoED44iBE+rGASPl0xEz6/+RM+TLkUPm9vFT6qGxY+"
        "aL0WPiNUFz5Q3xc+Zl4YPuDQGD4rNhk+wI0ZPhnXGT6jERo+1jwaPjIw7D0r3e09"
        "IZrvPfxl8T2uP/M9HSb1PTAY9z3MFPk95hr7PWQp/T0rP/89ka0APhe+AT6p0AI+"
        "rOQDPpr5BD7sDgY+FiQHPoU4CD6xSwk+FV0KPh1sCz5CeAw++YANPrqFDj7rhQ8+"
        "FIEQPpt2ET74ZRI+pU4TPhAwFD65CRU+BdsVPm2jFj5tYhc+dhcYPvbBGD5sYRk+"
        "RvUZPvx8Gj4C+Bo+xGUbPsTFGz5zFxw+PlocPqCNHD6Ove09dG/vPT4x8T21AfM9"
        "3N/0PZfK9j3cwPg9gsH6PXDL/D203f49hnsAPkOLAT59nQI+r7EDPkXHBD6/3QU+"
        "kPQGPiQLCD79IAk+jTUKPkdICz6XWAw+BmYNPv9vDj70dQ8+W3cQPrJzET5mahI+"
        "8VoTPsZEFD5YJxU+IQIWPpTUFj4knhc+S14YPoIUGT4wwBk+22AaPun1Gj7bfhs+"
        "I/sbPjVqHD6Eyxw+iR4dPrhiHT6Llx0+2VrvPVAR8T1L1/I9pKv0PU2N9j08e/g9"
        "Y3T6PZp3/D3kg/49GkwAPrBZAT43agI+GX0DPtiRBD7mpwU+vr4GPtnVBz6j7Ag+"
        "nQIKPjsXCz71KQw+PzoNPpJHDj5pUQ8+N1cQPm5YET6QVBI+DksTPl07FD72JBU+"
        "UgcWPuXhFj4wtBc+mH0YPqQ9GT7A8xk+bp8aPiBAGz5J1Rs+Y14cPufaHD5KSh0+"
        "/asdPon/HT5SRB4+2nkePqzE8T0ng/M9RFD1PQIr9z1jEvk9TAX7PcAC/T2+Cf89"
        "lowAPgCYAT6jpgI+7LcDPmLLBD5/4AU+vfYGPpQNCD6HJAk+BzsKPp1QCz60ZAw+"
        "1HYNPnaGDj4Ukw8+L5wQPjShET6voRI+FJ0TPt6SFD6LghU+kGsWPnRNFz6qJxg+"
        "sfkYPgXDGT4Xgxo+dzkbPonlGz7chhw+4xwdPh2nHT7/JB4+CJYePrP5Hj6ATx8+"
        "4ZYfPl/PHz662/U9a6j3PTeC+T1JaPs9hVn9PQhV/z3irAA+bLMBPqG9Aj4CywM+"
        "FtsEPl7tBT5hAQc+nxYIPqAsCT7eQgo+5VgLPj9uDD5cgg0+05QOPh2lDz7CshA+"
        "QL0RPiHEEj7kxhM+EcUUPii+FT6wsRY+Kp8XPhaGGD4DZhk+Yj4aPsoOGz661hs+"
        "rJUcPidLHT6z9h0+15cePg0uHz7huB8+zzcgPl6qID4UECE+bGghPvqyIT4x7yE+"
        "V43/PUqqAD4mlAE+04MCPs14Az6mcgQ+53AFPhBzBj61eAc+XoEIPoyMCT7LmQo+"
        "pKgLPqS4DD5MyQ0+MNoOPsvqDz6w+hA+aAkSPngWEz5pIRQ+yCkVPhYvFj7oMBc+"
        "ty4YPh4oGT6YHBo+qwsbPvP0Gz7k1xw+F7QdPg6JHj5KVh8+ZRsgPtnXID41iyE+"
        "BjUiPs3UIj4RaiM+ZvQjPlVzJD5V5iQ+BE0lPtqmJT5s8yU+QjImPkIKzj1Qk8M9"
        "QaG5PfkvsD1TO6c9Lb+ePWa3lj3ZH489ZvSHPegwgT1/onU9jqJpPcBZXj3Iv1M9"
        "ZcxJPVR3QD1LuDc9CYcvPUnbJz3GrCA9OfMZPV2mEz3xvQ09sDEIPU/5Aj0bGfw8"
        "UcbyPK7p6TyscuE8v1DZPGVz0TwJysk8LUTCPD3Rujy1YLM8BuKrPK5EpDwbeJw8"
        "zWuUPDIPjDzEUYM87UV0PHzkYDwyXkw87ZE2PKdeHzw="),
    "cld/abari": ("f4", (5,),
        "1/dhO9f3YTvX92E71/dhO9f3YTs="),
    "cld/bbari": ("f4", (5,),
        "gZUbQIGVG0CBlRtAgZUbQIGVG0A="),
    "cld/cbari": ("f4", (5,),
        "rMUnN82v5jhfKUs8sMkaPTLm7j4="),
    "cld/dbari": ("f4", (5,),
        "AAAAAEq4azeoAzQ65EuoOmr3qzc="),
    "cld/ebari": ("f4", (5,),
        "IR9EP1TjRT8QWEk/BTRRP8uhdT8="),
    "cld/fbari": ("f4", (5,),
        "ZWEZOiyBFDo62Tw6WKNDOk+n4Tg="),
    "ref/pref": ("f4", (59,),
        "KbSDRBepV0RokTBEvo8QRMm27EMUzsFDi6yeQ1jpgUOauVRD/CkuQxCYDkP0felC"
        "wSq/QqODnEKcJIBCRtRRQinLK0IepwxCSFDmQZeQvEFOYppB+8t8QQn5TkG8dClB"
        "/bwKQbYt40CC/7lAQUiYQC1beUDQJ0xAGCYnQJTZCED0FeA/XHe3P5M1lj9F9nU/"
        "dGBJP4vfJD+c/AY/GAndPg74tD4kKpQ+QZ1yPrGiRj7soCI+UiYFPqcG2j1dgbI9"
        "0iWSPa9Pbz2H7kM9U2ogPUZWAz0QD9c8WROwPKIokDzbDWw8l0NBPEo7Hjw="),
    "ref/preflog": ("f4", (59,),
        "UrjeQOxR2ECF69FAH4XLQLgexUBSuL5A7FG4QIXrsUAfhatAuB6lQFK4nkDsUZhA"
        "heuRQB+Fi0C4HoVApHB9QNejcEAK12NAPQpXQHE9SkCkcD1A16MwQArXI0A9ChdA"
        "cT0KQEjh+j+uR+E/FK7HP3sUrj/hepQ/j8J1P1yPQj8pXA8/7FG4PgrXIz4K1yO9"
        "j8J1vq5H4b4K1yO/PQpXv7gehb9SuJ6/7FG4v4Xr0b8fheu/XI8CwClcD8D2KBzA"
        "w/UowI/CNcBcj0LAKVxPwPYoXMDD9WjAj8J1wK5HgcAUrofAexSOwOF6lMA="),
    "ref/tref": ("f4", (59,),
        "mhmTQ7j+j0NSeItDAKCGQz3qgUOFK3pDH8VwQz3KZ0NcD19DrsdXQzOzV0Mzs1dD"
        "M7NXQ1wPWUN7lFpDFC5cQ3G9XUOuR19DPcpgQ82MYkMKV2RDSCFnQ48CakOuB21D"
        "UjhwQ8O1c0OPQndDmtl6Q+yRfkP2KIFDexSDQ64HhUO4/oZDM7OIQxSuiUMK14lD"
        "KdyIQ6TQh0NmxoZDCveEQxQOg0PXI4FDmpl9Qz2KeEPhenNDChduQ1wPaEOuB2JD"
        "AABcQ5pZVkO43lBDZmZLQ4XrRUNm5kBDChc8Qz1KN0NxfTJDpPAtQ7geLEM="),
    "aer/rsrtaua": ("f4", (14, 6),
        "ATDePQEw3j0+XFI+PlxSPj5cUj4+XFI+PlxSPulDBz/pQwc/EeTYPxHk2D8R5Ng/"
        "EeTYPwEw3j3bvyo/278qP/uuWD/7rlg/+65YP/uuWD/7rlg/Qs9uP0LPbj+lLI8/"
        "pSyPP6Usjz+lLI8/278qP4QNJz+EDSc/E35ZPxN+WT8Tflk/E35ZPxN+WT+9Om8/"
        "vTpvP5fKiz+Xyos/l8qLP5fKiz+EDSc/aJHtPWiR7T147l0+eO5dPnjuXT547l0+"
        "eO5dPjPhBz8z4Qc/eVjcP3lY3D95WNw/eVjcP2iR7T3DtoU9w7aFPQ6+kD4OvpA+"
        "Dr6QPg6+kD4OvpA+HeYrPx3mKz8w8IQ/MPCEPzDwhD8w8IQ/w7aFPWACNz1gAjc9"
        "CYrfPQmK3z0Jit89CYrfPQmK3z0Kou4+CqLuPpRqjz+Uao8/lGqPP5Rqjz9gAjc9"),
    "aer/rsrpiza": ("f4", (14, 6),
        "ouYFP6LmBT+gJlQ/oCZUP6AmVD+gJlQ/oCZUP6eiZT+nomU/RzZqP0c2aj9HNmo/"
        "RzZqP6LmBT8fb0k/H29JP120fj9dtH4/XbR+P120fj9dtH4/Tp1/P06dfz/G4H4/"
        "xuB+P8bgfj/G4H4/H29JPz5oWj8+aFo/ao5tP2qObT9qjm0/ao5tP2qObT/tr2w/"
        "7a9sPwseQD8LHkA/Cx5APwseQD8+aFo/5kPPPuZDzz5wLy0/cC8tP3AvLT9wLy0/"
        "cC8tP+PeRj/j3kY/hClQP4QpUD+EKVA/hClQP+ZDzz5o9F8/aPRfP7hecj+4XnI/"
        "uF5yP7hecj+4XnI/6gl0P+oJdD9TsHA/U7BwP1OwcD9TsHA/aPRfP2Y4cT5mOHE+"
        "PN9+Pzzffj88334/PN9+Pzzffj/+/38//v9/P/7/fz/+/38//v9/P/7/fz9mOHE+"),
    "aer/rsrasya": ("f4", (14, 6),
        "LVszPy1bMz9P5yI/T+ciP0/nIj9P5yI/T+ciP0seKz9LHis//aA6P/2gOj/9oDo/"
        "/aA6Py1bMz+IoVE/iKFRP3puTT96bk0/em5NP3puTT96bk0/Gt1JPxrdST/dmU0/"
        "3ZlNP92ZTT/dmU0/iKFRP2zQMz9s0DM/XfkwP135MD9d+TA/XfkwP135MD/T3DI/"
        "09wyPwXbSD8F20g/BdtIPwXbSD9s0DM/XpwwP16cMD+loyA/paMgP6WjID+loyA/"
        "paMgP89MKD/PTCg/Q1M2P0NTNj9DUzY/Q1M2P16cMD+UDO0+lAztPqVOHD+lThw/"
        "pU4cP6VOHD+lThw/sGssP7BrLD9DaTM/Q2kzP0NpMz9DaTM/lAztPp1XQz6dV0M+"
        "rsDzPq7A8z6uwPM+rsDzPq7A8z6M5yY/jOcmP0MgOj9DIDo/QyA6P0MgOj+dV0M+"),
    "wvn/wavenum1": ("f4", (14,),
        "AIAiRQAgS0UAAHpFAFCRRQDwoEUAMMBFAKDwRQCQ+0UAyEhGAAB6RgD0sEYAkOJG"
        "AHAURwAATUQ="),
    "wvn/wavenum2": ("f4", (14,),
        "ACBLRQAAekUAUJFFAPCgRQAwwEUAoPBFAJD7RQDISEYAAHpGAPSwRgCQ4kYAcBRH"
        "AFBDRwCAIkU="),
    "wvn/ngb": ("i4", (112,),
        "EAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEQAAABEAAAARAAAAEQAAABEAAAARAAAA"
        "EQAAABEAAAARAAAAEQAAABEAAAARAAAAEgAAABIAAAASAAAAEgAAABIAAAASAAAA"
        "EgAAABIAAAATAAAAEwAAABMAAAATAAAAEwAAABMAAAATAAAAEwAAABQAAAAUAAAA"
        "FAAAABQAAAAUAAAAFAAAABQAAAAUAAAAFAAAABQAAAAVAAAAFQAAABUAAAAVAAAA"
        "FQAAABUAAAAVAAAAFQAAABUAAAAVAAAAFgAAABYAAAAXAAAAFwAAABcAAAAXAAAA"
        "FwAAABcAAAAXAAAAFwAAABcAAAAXAAAAGAAAABgAAAAYAAAAGAAAABgAAAAYAAAA"
        "GAAAABgAAAAZAAAAGQAAABkAAAAZAAAAGQAAABkAAAAaAAAAGgAAABoAAAAaAAAA"
        "GgAAABoAAAAbAAAAGwAAABsAAAAbAAAAGwAAABsAAAAbAAAAGwAAABwAAAAcAAAA"
        "HAAAABwAAAAcAAAAHAAAAB0AAAAdAAAAHQAAAB0AAAAdAAAAHQAAAB0AAAAdAAAA"
        "HQAAAB0AAAAdAAAAHQAAAA=="),
}
# --- END GENERATED STATIC TABLES ---


# ===========================================================================
# SECTION 8: init-time table builds and production table assembly
#
# The band coefficient tables come from woof.ingest.rrtmg_coeffs (sibling
# contract; raw file image plus the bit-exact cmbgb reduction).  What that
# loader deliberately does not cover - because it is not in RRTMG_SW_DATA -
# is rebuilt or embedded here:
#   * exp_tbl / bpade  (built by rrtmg_sw_ini at model start; rebuilt here
#     with the glibc expf transcription and gated bitwise against the
#     oracle dump),
#   * heatfac / oneminus (swdatinit formulas),
#   * the swcldpr / swatmref / swaerpr / wavenumber DATA tables (embedded
#     below as generated base64 of the verified oracle dump).
# ===========================================================================

_NTBL = 10000


def build_exp_tbl():
    """rrtmg_sw_ini's exponential lookup table, FP32, glibc expf."""
    pade = F(0.278)
    bpade = F(F(1.0) / pade)
    expeps = F(1.0e-20)
    exp_tbl = np.zeros(_NTBL + 1, dtype=np.float32)
    exp_tbl[0] = F(1.0)
    exp_tbl[_NTBL] = expeps
    for itr in range(1, _NTBL):
        tfn = F(F(float(itr)) / F(float(_NTBL)))
        tau_tbl = F(F(bpade * tfn) / F(F(1.0) - tfn))
        v = expf(F(-tau_tbl))
        if v <= expeps:
            v = expeps
        exp_tbl[itr] = v
    return exp_tbl, bpade


def build_heatfac(cpdair=F(1004.5)):
    """swdatinit: heatfac = grav * secdy / (cpdair * 1.e2), FP32.

    WRF calls rrtmg_sw_ini(cp) with cp = 7*r_d/2 = 1004.5 from
    module_model_constants.
    """
    grav = F(9.8066)
    secdy = F(8.6400e4)
    return F(F(grav * secdy) / F(F(cpdair) * F(1.0e2)))


def _static(name):
    dt, shape, b64 = _STATIC_TABLES[name]
    buf = __import__("base64").b64decode(b64)
    a = np.frombuffer(buf, dtype=np.dtype("<" + dt))
    return a.reshape(shape, order="F").astype(dt, copy=False)


def _fill_common(t: SWTables):
    t.exp_tbl, t.bpade = build_exp_tbl()
    t.od_lo = F(0.06)
    t.tblint = F(10000.0)
    t.heatfac = build_heatfac()
    t.oneminus = F(F(1.0) - F(1.0e-06))
    t.grav = F(9.8066)
    t.avogad = F(6.02214199e23)
    t.pref = _static("ref/pref")
    t.preflog = _static("ref/preflog")
    t.tref = _static("ref/tref")
    for name in ("extliq1", "ssaliq1", "asyliq1", "extice2", "ssaice2",
                 "asyice2", "extice3", "ssaice3", "asyice3", "fdlice3",
                 "abari", "bbari", "cbari", "dbari", "ebari", "fbari"):
        setattr(t, name, _static(f"cld/{name}"))
    t.ngb = _static("wvn/ngb")
    t.wavenum1 = _static("wvn/wavenum1")
    t.wavenum2 = _static("wvn/wavenum2")
    t.rsrtaua = _static("aer/rsrtaua")
    t.rsrpiza = _static("aer/rsrpiza")
    t.rsrasya = _static("aer/rsrasya")


def tables_from_coeffs(coeffs) -> SWTables:
    """Assemble :class:`SWTables` from the ingest loader's frozen-contract
    dict (``load_rrtmg_sw_coefficients()``), plus the init builds and
    embedded static tables above."""
    t = SWTables()
    t.kg = {band: dict(coeffs[f"rrsw_kg{band}"]) for band in range(16, 30)}
    _fill_common(t)
    return t


#: The aer_opt=1 climatology, ``woof/data``-relative.  It ships in the
#: recast-woof-data companion (woof.data_assets.COMPANION_TREES): at 6.0 MB it
#: was what put the 2.9.0 platform wheel at 101,425,913 bytes, over PyPI's
#: 100,000,000-byte per-file limit.
AER1_CLIMATOLOGY_ASSET = "wrf_radiation/rrtmg_aer1_climatology.npz"


def encode_aer1_climatology(climatology) -> np.ndarray:
    """The climatology's float32 words, stored losslessly and compactly.

    Each word's bits as little-endian uint32, differenced along the last
    (longitude) axis modulo 2**32, the four bytes of every word split into
    four planes, then xz.  6,027,855 bytes as plain deflate become about 4.6
    MB, which is what lets the companion wheel carry it under PyPI's per-file
    limit.  :func:`decode_aer1_climatology` inverts it exactly; the
    manifest's sha256 of the decoded words is what the loader checks.
    """
    import lzma

    words = np.ascontiguousarray(climatology, dtype="<f4").view("<u4")
    delta = np.diff(words, axis=-1, prepend=np.zeros(
        words.shape[:-1] + (1,), dtype="<u4")).astype("<u4")
    planes = delta.reshape(-1).view(np.uint8).reshape(-1, 4).T.tobytes()
    return np.frombuffer(
        lzma.compress(planes, preset=9 | lzma.PRESET_EXTREME), dtype=np.uint8)


def decode_aer1_climatology(payload, shape) -> np.ndarray:
    """Inverse of :func:`encode_aer1_climatology`: the float32 climatology."""
    import lzma

    shape = tuple(int(n) for n in shape)
    raw = np.frombuffer(lzma.decompress(np.asarray(payload, np.uint8).tobytes()),
                        dtype=np.uint8)
    if raw.size != 4 * int(np.prod(shape)):
        raise ValueError("aer_opt=1 climatology payload has the wrong length for its shape")
    delta = np.ascontiguousarray(raw.reshape(4, -1).T).view("<u4").reshape(shape)
    return np.cumsum(delta, axis=-1, dtype="<u4").view("<f4")


def load_sw_tables(path=None, *, aer_opt=0, data_dir=None) -> SWTables:
    """Production entry: load coefficients via woof.ingest.rrtmg_coeffs
    (imported lazily) and assemble tables.
    """
    from woof.ingest.rrtmg_coeffs import load_rrtmg_sw_coefficients
    tables = tables_from_coeffs(load_rrtmg_sw_coefficients(path))
    if aer_opt == 1:
        from woof.ingest.rrtmg_coeffs import DATA_DIR
        import hashlib
        import json
        from pathlib import Path
        root = DATA_DIR if data_dir is None else Path(data_dir)
        from woof.tables import table_path
        manifest = json.loads(table_path("rrtmg-aer1", root / "rrtmg_aer1_manifest.json").read_text())
        if data_dir is None:
            from woof import data_assets
            asset = data_assets.data_path(AER1_CLIMATOLOGY_ASSET)
        else:
            asset = root / Path(AER1_CLIMATOLOGY_ASSET).name
        from woof.tables import CorruptTableAssets
        try:
            asset = table_path("rrtmg-aer1", asset)
        except CorruptTableAssets as error:
            raise CorruptTableAssets(f"aer_opt=1 climatology manifest hash mismatch: {error}") from error
        if asset.stat().st_size != manifest["asset_size"] or hashlib.sha256(asset.read_bytes()).hexdigest() != manifest["asset_sha256"]:
            raise ValueError("aer_opt=1 climatology manifest hash mismatch; altered aerosol values would change AODTOT and shortwave")
        with np.load(asset, allow_pickle=False) as source:
            tables.aer1_climatology = {
                name: (decode_aer1_climatology(source["climatology_xz"],
                                               manifest["arrays"][name]["shape"])
                       if name == "climatology" else source[name])
                for name in manifest["arrays"]}
        for name, pin in manifest["arrays"].items():
            value = tables.aer1_climatology[name]
            if list(value.shape) != pin["shape"] or str(value.dtype) != pin["dtype"] or hashlib.sha256(value.tobytes()).hexdigest() != pin["sha256"]:
                raise ValueError(f"aer_opt=1 {name} array manifest mismatch")
        tables.aer1_optics_sha256 = manifest["optics_sha256"]
        validate_aer1_optics_tables(tables)
    return tables


# ===========================================================================
# SECTION 9: inatm_sw, the rrtmg_sw composition, and the WRF option-4
# driver surface (solar constant / zenith handling, output mapping)
# ===========================================================================

AMD = F(28.9660)      # molecular weight of dry air (inatm_sw locals)
AMW = F(18.0160)      # molecular weight of water vapor


def earth_sun(idn):
    """WRF option 4 never evaluates earth_sun: RRTMG_SWRAD hardcodes
    dyofyr = 0 / adjes = 1.0 because WRF's solcon already carries the
    eccentricity adjustment (and scon = solcon*(1-obscur)).  A bit-exact
    port would need glibc sinf/cosf transcriptions that nothing exercises;
    failing closed is safer than shipping an unverified trig path."""
    raise NotImplementedError(
        "earth_sun is unreachable via WRF option 4 (dyofyr = 0); "
        "no verified FP32 sinf/cosf transcription is provided")


def option4_trace_gases(yr, overrides=None):
    """The ghg_input = 0 trace-gas values of RRTMG_SWRAD, with WRF's exact
    mixed precision: the co2 expression is evaluated in REAL(4) (including
    a REAL(4) exp) and only then widened to the REAL(8) local.

    Returns (co2vmr, ch4vmr, n2ovmr, o2vmr) as float32.
    """
    co2 = F(F(F(280.0) + F(F(90.0) * expf(F(F(0.02) * (int(yr) - 2000))))) *
            F(1.0e-6))
    values = (co2, F(1774.0e-9), F(319.0e-9), F(0.209488))
    if overrides is None:
        return values
    from woof.core.trace_gases import LEGACY_SW_GASES, validate_trace_gas_overrides
    selected = validate_trace_gas_overrides(
        overrides, supported=LEGACY_SW_GASES, consumer="legacy RRTMG shortwave")
    return tuple(F(selected[name]) if name in selected else value
                 for name, value in zip(("co2", "ch4", "n2o", "o2"), values))


def inatm_sw(tab: SWTables, nlay, icld, iaer, play, plev, tlay, tlev, tsfc,
             h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
             adjes, dyofyr, scon, inflgsw, iceflgsw, liqflgsw,
             cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl,
             ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl,
             tauaer, ssaaer, asmaer):
    """Port of inatm_sw for one column.  Layer arrays are 1-D (nlay,),
    level arrays (nlay+1,), McICA arrays (ngptsw, nlay), aerosol arrays
    (nlay, nbndsw).  Returns the Fortran outputs as a dict; adjflux/solvar
    carry bands 16..29 as a length-14 array (index jb-16)."""
    one = F(1.0)
    nlayers = int(nlay)

    wkl = np.zeros((MXMOL, nlayers), dtype=np.float32)
    cldfmc = np.zeros((NGPTSW, nlayers), dtype=np.float32)
    taucmc = np.zeros((NGPTSW, nlayers), dtype=np.float32)
    ssacmc = np.ones((NGPTSW, nlayers), dtype=np.float32)
    asmcmc = np.zeros((NGPTSW, nlayers), dtype=np.float32)
    fsfcmc = np.zeros((NGPTSW, nlayers), dtype=np.float32)
    ciwpmc = np.zeros((NGPTSW, nlayers), dtype=np.float32)
    clwpmc = np.zeros((NGPTSW, nlayers), dtype=np.float32)
    cswpmc = np.zeros((NGPTSW, nlayers), dtype=np.float32)
    reicmc = np.zeros(nlayers, dtype=np.float32)
    relqmc = np.zeros(nlayers, dtype=np.float32)
    resnmc = np.zeros(nlayers, dtype=np.float32)
    taua = np.zeros((nlayers, NBNDSW), dtype=np.float32)
    ssaa = np.ones((nlayers, NBNDSW), dtype=np.float32)
    asma = np.zeros((nlayers, NBNDSW), dtype=np.float32)

    adjflx = F(adjes)
    if int(dyofyr) > 0:
        earth_sun(dyofyr)   # raises: unreachable via WRF option 4

    solvar = np.zeros(NBNDSW, dtype=np.float32)
    adjflux = np.zeros(NBNDSW, dtype=np.float32)
    for ib in range(NBNDSW):
        solvar[ib] = F(F(scon) / RRSW_SCON)
        adjflux[ib] = F(adjflx * solvar[ib])

    tbound = F(tsfc)

    pavel = np.zeros(nlayers, dtype=np.float32)
    tavel = np.zeros(nlayers, dtype=np.float32)
    pz = np.zeros(nlayers + 1, dtype=np.float32)
    tz = np.zeros(nlayers + 1, dtype=np.float32)
    pdp = np.zeros(nlayers, dtype=np.float32)
    coldry = np.zeros(nlayers, dtype=np.float32)

    pz[0] = F(plev[0])
    tz[0] = F(tlev[0])
    for l in range(1, nlayers + 1):
        pavel[l - 1] = play[l - 1]
        tavel[l - 1] = tlay[l - 1]
        pz[l] = plev[l]
        tz[l] = tlev[l]
        pdp[l - 1] = F(pz[l - 1] - pz[l])
        wkl[0, l - 1] = h2ovmr[l - 1]
        wkl[1, l - 1] = co2vmr[l - 1]
        wkl[2, l - 1] = o3vmr[l - 1]
        wkl[3, l - 1] = n2ovmr[l - 1]
        wkl[5, l - 1] = ch4vmr[l - 1]
        wkl[6, l - 1] = o2vmr[l - 1]
        amm = F(F(F(one - wkl[0, l - 1]) * AMD) + F(wkl[0, l - 1] * AMW))
        coldry[l - 1] = F(F(F(F(pz[l - 1] - pz[l]) * F(1.0e3)) * tab.avogad) /
                          F(F(F(F(1.0e2) * tab.grav) * amm) *
                            F(one + wkl[0, l - 1])))

    for l in range(nlayers):
        for imol in range(NMOL):
            wkl[imol, l] = F(coldry[l] * wkl[imol, l])

    if iaer >= 1:
        for ib in range(NBNDSW):
            for l in range(nlayers):
                taua[l, ib] = tauaer[l, ib]
                ssaa[l, ib] = ssaaer[l, ib]
                asma[l, ib] = asmaer[l, ib]

    inflag = iceflag = liqflag = 0
    if icld >= 1:
        inflag = int(inflgsw)
        iceflag = int(iceflgsw)
        liqflag = int(liqflgsw)
        for l in range(nlayers):
            cldfmc[:, l] = cldfmcl[:, l]
            taucmc[:, l] = taucmcl[:, l]
            ssacmc[:, l] = ssacmcl[:, l]
            asmcmc[:, l] = asmcmcl[:, l]
            fsfcmc[:, l] = fsfcmcl[:, l]
            ciwpmc[:, l] = ciwpmcl[:, l]
            clwpmc[:, l] = clwpmcl[:, l]
            if iceflag == 5:
                cswpmc[:, l] = cswpmcl[:, l]
            reicmc[l] = reicmcl[l]
            relqmc[l] = relqmcl[l]
            if iceflag == 5:
                resnmc[l] = resnmcl[l]

    return dict(nlayers=nlayers, pavel=pavel, pz=pz, pdp=pdp, tavel=tavel,
                tz=tz, tbound=tbound, coldry=coldry, wkl=wkl,
                adjflux=adjflux, solvar=solvar, inflag=inflag,
                iceflag=iceflag, liqflag=liqflag, cldfmc=cldfmc,
                taucmc=taucmc, ssacmc=ssacmc, asmcmc=asmcmc, fsfcmc=fsfcmc,
                ciwpmc=ciwpmc, clwpmc=clwpmc, cswpmc=cswpmc, reicmc=reicmc,
                relqmc=relqmc, resnmc=resnmc, taua=taua, ssaa=ssaa,
                asma=asma)


def rrtmg_sw(tab: SWTables, nlay, icld, play, plev, tlay, tlev, tsfc,
             h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
             asdir, asdif, aldir, aldif, coszen, adjes, dyofyr, scon,
             inflgsw, iceflgsw, liqflgsw,
             cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl,
             ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl,
             tauaer, ssaaer, asmaer, aer_opt=0):
    """Port of the rrtmg_sw composition for one column (ncol = 1, the only
    way WRF option 4 calls it).  Fails closed on aer_opt = 1 (the iaer = 6
    ECMWF-aerosol branch; the campaign and every fixture run aer_opt = 0)
    and on dyofyr > 0 (see :func:`earth_sun`).

    Returns a dict with swuflx/swdflx/swhr (total sky), swuflxc/swdflxc/
    swhrc (clear sky), sibvisdir/sibvisdif/sibnirdir/sibnirdif,
    swdkdir/swdkdif/swdkdirc, and swuflxcln/swdflxcln (zeros; WRF_CHEM=0),
    all bottom-to-top like the Fortran.
    """
    zepzen = F(1.0e-10)
    if aer_opt in (0, 2, 3):
        iaer = 10
    else:
        raise NotImplementedError(
            "aer_opt = 1 (iaer = 6 ECMWF aerosol path) is not fixture-"
            "verified; the campaign runs aer_opt = 0")

    ia = inatm_sw(tab, nlay, icld, iaer, play, plev, tlay, tlev, tsfc,
                  h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
                  adjes, dyofyr, scon, inflgsw, iceflgsw, liqflgsw,
                  cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl,
                  ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl,
                  tauaer, ssaaer, asmaer)
    nlayers = ia["nlayers"]

    taormc = cldprmc_sw(tab, nlayers, ia["inflag"], ia["iceflag"],
                        ia["liqflag"], ia["cldfmc"], ia["ciwpmc"],
                        ia["clwpmc"], ia["cswpmc"], ia["reicmc"],
                        ia["relqmc"], ia["resnmc"], ia["taucmc"],
                        ia["ssacmc"], ia["asmcmc"], ia["fsfcmc"])

    sc = setcoef_sw(tab, nlayers, ia["pavel"], ia["tavel"], ia["coldry"],
                    ia["wkl"])

    cossza = F(coszen)
    if cossza <= zepzen:
        cossza = zepzen

    albdir = np.zeros(NBNDSW, dtype=np.float32)
    albdif = np.zeros(NBNDSW, dtype=np.float32)
    for ib in range(1, 10):
        albdir[ib - 1] = F(aldir)
        albdif[ib - 1] = F(aldif)
    albdir[NBNDSW - 1] = F(aldir)
    albdif[NBNDSW - 1] = F(aldif)
    for ib in range(10, 14):
        albdir[ib - 1] = F(asdir)
        albdif[ib - 1] = F(asdif)

    zcldfmc = np.zeros((nlayers, NGPTSW), dtype=np.float32)
    ztaucmc = np.zeros((nlayers, NGPTSW), dtype=np.float32)
    ztaormc = np.zeros((nlayers, NGPTSW), dtype=np.float32)
    zasycmc = np.zeros((nlayers, NGPTSW), dtype=np.float32)
    zomgcmc = np.ones((nlayers, NGPTSW), dtype=np.float32)
    if icld >= 1:
        zcldfmc[:, :] = ia["cldfmc"].T
        ztaucmc[:, :] = ia["taucmc"].T
        ztaormc[:, :] = taormc.T
        zasycmc[:, :] = ia["asmcmc"].T
        zomgcmc[:, :] = ia["ssacmc"].T

    # iaer == 10 (aerosol arrays pass through; zeros in the campaign)
    ztaua = ia["taua"].copy()
    zasya = ia["asma"].copy()
    zomga = ia["ssaa"].copy()

    out = spcvmc_sw(tab, nlayers, JPB1, JPB2, 1,
                    albdif, albdir, zcldfmc, ztaucmc, zasycmc, zomgcmc,
                    ztaormc, ztaua, zasya, zomga, cossza,
                    ia["adjflux"], sc["laytrop"], sc["jp"], sc["jt"],
                    sc["jt1"], sc["colch4"], sc["colco2"], sc["colh2o"],
                    sc["colmol"], sc["colo2"], sc["colo3"],
                    sc["fac00"], sc["fac01"], sc["fac10"], sc["fac11"],
                    sc["selffac"], sc["selffrac"], sc["indself"],
                    sc["forfac"], sc["forfrac"], sc["indfor"])

    nl1 = nlayers + 1
    swuflxc = out["pbbcu"][:nl1].copy()
    swdflxc = out["pbbcd"][:nl1].copy()
    swuflx = out["pbbfu"][:nl1].copy()
    swdflx = out["pbbfd"][:nl1].copy()
    dirdflux = out["pbbfddir"][:nl1].copy()
    swdkdir = dirdflux.copy()
    swdkdif = np.zeros(nl1, dtype=np.float32)
    sibvisdir = np.zeros(nl1, dtype=np.float32)
    sibvisdif = np.zeros(nl1, dtype=np.float32)
    sibnirdir = np.zeros(nl1, dtype=np.float32)
    sibnirdif = np.zeros(nl1, dtype=np.float32)
    swdkdirc = out["pbbcddir"][:nl1].copy()
    for i in range(nl1):
        swdkdif[i] = F(swdflx[i] - dirdflux[i])
        sibvisdir[i] = out["puvfddir"][i]
        sibvisdif[i] = F(out["puvfd"][i] - out["puvfddir"][i])
        sibnirdir[i] = out["pnifddir"][i]
        sibnirdif[i] = F(out["pnifd"][i] - out["pnifddir"][i])

    swnflxc = np.zeros(nl1, dtype=np.float32)
    swnflx = np.zeros(nl1, dtype=np.float32)
    for i in range(nl1):
        swnflxc[i] = F(swdflxc[i] - swuflxc[i])
        swnflx[i] = F(swdflx[i] - swuflx[i])
    swhrc = np.zeros(nlayers, dtype=np.float32)
    swhr = np.zeros(nlayers, dtype=np.float32)
    for i in range(nlayers - 1):
        zdpgcp = F(tab.heatfac / ia["pdp"][i])
        swhrc[i] = F(F(swnflxc[i + 1] - swnflxc[i]) * zdpgcp)
        swhr[i] = F(F(swnflx[i + 1] - swnflx[i]) * zdpgcp)
    swhrc[nlayers - 1] = F(0.0)
    swhr[nlayers - 1] = F(0.0)

    return dict(swuflx=swuflx, swdflx=swdflx, swhr=swhr, swuflxc=swuflxc,
                swdflxc=swdflxc, swhrc=swhrc,
                swuflxcln=np.zeros(nl1, dtype=np.float32),
                swdflxcln=np.zeros(nl1, dtype=np.float32),
                sibvisdir=sibvisdir, sibvisdif=sibvisdif,
                sibnirdir=sibnirdir, sibnirdif=sibnirdif,
                swdkdir=swdkdir, swdkdif=swdkdif, swdkdirc=swdkdirc,
                spc=out, inatm=ia, setcoef=sc)


def swrad_option4_outputs(res, pi3d, coszrs, kte):
    """The WRF-level output mapping of RRTMG_SWRAD's day branch, from the
    :func:`rrtmg_sw` result for one column.  pi3d is the Exner function on
    layers (kts..kte); coszrs the unclamped xcoszen (> 0 by the driver's
    night gate).  kte = nlay - 1 (WRF adds the extra TOA layer).

    Night behavior (coszrs <= 0), for the integration wave -- WRAPPER
    level, per the SW audit: RRTMG_SWRAD skips the SW call entirely,
    still writes COSZR = xcoszen, zeroes ONLY SWCF plus its listed
    scalar diagnostics, and leaves GSW, the flux profiles, and
    RTHRATENSW UNTOUCHED (see swrad_night_outputs / SW_NIGHT_ZEROED /
    SW_NIGHT_UNTOUCHED in rrtmg_legacy_prep, which implement exactly
    that).  Separately, WRF's radiation DRIVER zeroes GSW and
    RTHRATENSW for the whole grid at the top of every radiation call
    (module_radiation_driver.F:1721,1738), so a night column ends a
    radiation step with zeros through driver-zero + wrapper-skip -- but
    that zeroing belongs to the driver, not to this wrapper mapping.
    """
    swuflx, swdflx = res["swuflx"], res["swdflx"]
    swuflxc, swdflxc = res["swuflxc"], res["swdflxc"]
    o = {}
    o["gsw"] = F(swdflx[0] - swuflx[0])
    o["swcf"] = F(F(swdflx[kte + 1] - swuflx[kte + 1]) -
                  F(swdflxc[kte + 1] - swuflxc[kte + 1]))
    o["swupt"] = swuflx[kte + 1]
    o["swuptc"] = swuflxc[kte + 1]
    o["swdnt"] = swdflx[kte + 1]
    o["swdntc"] = swdflxc[kte + 1]
    o["swupb"] = swuflx[0]
    o["swupbc"] = swuflxc[0]
    o["swdnb"] = swdflx[0]
    o["swdnbc"] = swdflxc[0]
    o["swvisdir"] = res["sibvisdir"][0]
    o["swvisdif"] = res["sibvisdif"][0]
    o["swnirdir"] = res["sibnirdir"][0]
    o["swnirdif"] = res["sibnirdif"][0]
    o["swddir"] = res["swdkdir"][0]
    o["swddni"] = F(o["swddir"] / F(coszrs))
    o["swddif"] = res["swdkdif"][0]
    o["swdownc"] = swdflxc[0]
    o["swddirc"] = res["swdkdirc"][0]
    o["swddnic"] = F(o["swddirc"] / F(coszrs))
    # RTHRATENSW covers the WRF layers kts..kte only (the extra TOA layer
    # has no WRF-side theta tendency).
    rthratensw = np.zeros(kte, dtype=np.float32)
    rthratenswc = np.zeros(kte, dtype=np.float32)
    for k in range(kte):
        tten = F(res["swhr"][k] / F(86400.0))
        rthratensw[k] = F(tten / F(pi3d[k]))
        tten = F(res["swhrc"][k] / F(86400.0))
        rthratenswc[k] = F(tten / F(pi3d[k]))
    o["rthratensw"] = rthratensw
    o["rthratenswc"] = rthratenswc
    o["swupflx"] = swuflx
    o["swupflxc"] = swuflxc
    o["swdnflx"] = swdflx
    o["swdnflxc"] = swdflxc
    return o


# ===========================================================================
# SECTION 10: CUDA twin host driver
#
# Compiles woof/core/kernels/rrtmg_sw.cu with a generated preamble:
# exact-bit scalar #defines (via __uint_as_float) and offsets into one
# packed FP32 table buffer.  Data movement (copies/transposes) uses CuPy;
# every FP operation runs in the kernels through __f*_rn intrinsics.
# ===========================================================================

_CUDA_TABLE_ARRAYS = [
    # (define name, band, var)  - packed in this order, F-order flattened
    ("KG16_ABSA", 16, "absa"), ("KG16_ABSB", 16, "absb"),
    ("KG16_SELFREF", 16, "selfref"), ("KG16_FORREF", 16, "forref"),
    ("KG16_SFLUXREF", 16, "sfluxref"),
    ("KG17_ABSA", 17, "absa"), ("KG17_ABSB", 17, "absb"),
    ("KG17_SELFREF", 17, "selfref"), ("KG17_FORREF", 17, "forref"),
    ("KG17_SFLUXREF", 17, "sfluxref"),
    ("KG18_ABSA", 18, "absa"), ("KG18_ABSB", 18, "absb"),
    ("KG18_SELFREF", 18, "selfref"), ("KG18_FORREF", 18, "forref"),
    ("KG18_SFLUXREF", 18, "sfluxref"),
    ("KG19_ABSA", 19, "absa"), ("KG19_ABSB", 19, "absb"),
    ("KG19_SELFREF", 19, "selfref"), ("KG19_FORREF", 19, "forref"),
    ("KG19_SFLUXREF", 19, "sfluxref"),
    ("KG20_ABSA", 20, "absa"), ("KG20_ABSB", 20, "absb"),
    ("KG20_SELFREF", 20, "selfref"), ("KG20_FORREF", 20, "forref"),
    ("KG20_SFLUXREF", 20, "sfluxref"), ("KG20_ABSCH4", 20, "absch4"),
    ("KG21_ABSA", 21, "absa"), ("KG21_ABSB", 21, "absb"),
    ("KG21_SELFREF", 21, "selfref"), ("KG21_FORREF", 21, "forref"),
    ("KG21_SFLUXREF", 21, "sfluxref"),
    ("KG22_ABSA", 22, "absa"), ("KG22_ABSB", 22, "absb"),
    ("KG22_SELFREF", 22, "selfref"), ("KG22_FORREF", 22, "forref"),
    ("KG22_SFLUXREF", 22, "sfluxref"),
    ("KG23_ABSA", 23, "absa"),
    ("KG23_SELFREF", 23, "selfref"), ("KG23_FORREF", 23, "forref"),
    ("KG23_SFLUXREF", 23, "sfluxref"), ("KG23_RAYL", 23, "rayl"),
    ("KG24_ABSA", 24, "absa"), ("KG24_ABSB", 24, "absb"),
    ("KG24_SELFREF", 24, "selfref"), ("KG24_FORREF", 24, "forref"),
    ("KG24_SFLUXREF", 24, "sfluxref"),
    ("KG24_ABSO3A", 24, "abso3a"), ("KG24_ABSO3B", 24, "abso3b"),
    ("KG24_RAYLA", 24, "rayla"), ("KG24_RAYLB", 24, "raylb"),
    ("KG25_ABSA", 25, "absa"), ("KG25_SFLUXREF", 25, "sfluxref"),
    ("KG25_ABSO3A", 25, "abso3a"), ("KG25_ABSO3B", 25, "abso3b"),
    ("KG25_RAYL", 25, "rayl"),
    ("KG26_SFLUXREF", 26, "sfluxref"), ("KG26_RAYL", 26, "rayl"),
    ("KG27_ABSA", 27, "absa"), ("KG27_ABSB", 27, "absb"),
    ("KG27_SFLUXREF", 27, "sfluxref"), ("KG27_RAYL", 27, "rayl"),
    ("KG28_ABSA", 28, "absa"), ("KG28_ABSB", 28, "absb"),
    ("KG28_SFLUXREF", 28, "sfluxref"),
    ("KG29_ABSA", 29, "absa"), ("KG29_ABSB", 29, "absb"),
    ("KG29_SELFREF", 29, "selfref"), ("KG29_FORREF", 29, "forref"),
    ("KG29_SFLUXREF", 29, "sfluxref"),
    ("KG29_ABSH2O", 29, "absh2o"), ("KG29_ABSCO2", 29, "absco2"),
]

_CUDA_SCALARS = [
    ("KG16_RAYL", 16, "rayl"), ("KG16_STRRAT1", 16, "strrat1"),
    ("KG17_RAYL", 17, "rayl"), ("KG17_STRRAT", 17, "strrat"),
    ("KG18_RAYL", 18, "rayl"), ("KG18_STRRAT", 18, "strrat"),
    ("KG19_RAYL", 19, "rayl"), ("KG19_STRRAT", 19, "strrat"),
    ("KG20_RAYL", 20, "rayl"),
    ("KG21_RAYL", 21, "rayl"), ("KG21_STRRAT", 21, "strrat"),
    ("KG22_RAYL", 22, "rayl"), ("KG22_STRRAT", 22, "strrat"),
    ("KG23_GIVFAC", 23, "givfac"),
    ("KG24_STRRAT", 24, "strrat"),
    ("KG27_SCALEKUR", 27, "scalekur"),
    ("KG28_RAYL", 28, "rayl"), ("KG28_STRRAT", 28, "strrat"),
    ("KG29_RAYL", 29, "rayl"),
]


def _pack_cuda_tables(tab: SWTables):
    """Build (packed_f32_vector, defines_text) for the CUDA unit."""
    chunks = []
    lines = []
    off = 0
    for define, band, var in _CUDA_TABLE_ARRAYS:
        a = np.asarray(tab.kg[band][var], dtype=np.float32)
        flat = np.asfortranarray(a).reshape(-1, order="F")
        lines.append(f"#define RSW_T_{define} {off}")
        chunks.append(flat)
        off += flat.size
    for define, src in (("CLD_EXTLIQ1", tab.extliq1),
                        ("CLD_SSALIQ1", tab.ssaliq1),
                        ("CLD_ASYLIQ1", tab.asyliq1),
                        ("CLD_EXTICE2", tab.extice2),
                        ("CLD_SSAICE2", tab.ssaice2),
                        ("CLD_ASYICE2", tab.asyice2),
                        ("CLD_EXTICE3", tab.extice3),
                        ("CLD_SSAICE3", tab.ssaice3),
                        ("CLD_ASYICE3", tab.asyice3),
                        ("CLD_FDLICE3", tab.fdlice3),
                        ("REF_PREFLOG", tab.preflog),
                        ("REF_TREF", tab.tref),
                        ("EXP_TBL", tab.exp_tbl)):
        flat = np.asfortranarray(np.asarray(src, dtype=np.float32))
        flat = flat.reshape(-1, order="F")
        lines.append(f"#define RSW_T_{define} {off}")
        chunks.append(flat)
        off += flat.size
    for define, band, var in _CUDA_SCALARS:
        bits = np.asarray(tab.kg[band][var], dtype=np.float32).reshape(())
        u = int(bits.view(np.uint32))
        lines.append(f"#define RSW_C_{define} __uint_as_float({u}u)")
    for define, val in (("ONEMINUS", tab.oneminus), ("OD_LO", tab.od_lo),
                        ("BPADE", tab.bpade), ("TBLINT", tab.tblint)):
        u = int(np.float32(val).view(np.uint32))
        lines.append(f"#define RSW_C_{define} __uint_as_float({u}u)")
    return np.concatenate(chunks), "\n".join(lines) + "\n"


class CudaSW:
    """Compiled CUDA twin bound to one :class:`SWTables`."""

    def __init__(self, tab: SWTables):
        import cupy as cp
        self.cp = cp
        self.tab = tab
        packed, defines = _pack_cuda_tables(tab)
        src_path = __import__("pathlib").Path(__file__).parent / "kernels" / "rrtmg_sw.cu"
        code = defines + src_path.read_text(encoding="ascii")
        # CuPy appends -ftz=true to NVRTC options; that reaches even
        # mul.rn.f32 from __fmul_rn and flushes the subnormal
        # transmittance products (1e-20 * 1e-20 = 1e-40) the Fortran
        # keeps.  An explicit --ftz=false USED to take precedence, because
        # NVRTC honoured the last occurrence of a repeated option.  It no
        # longer does: NVRTC 13.0 REJECTS the duplicate outright --
        # ``nvrtc: error: --ftz (-ftz) defined more than once`` -- so
        # cupy.RawModule cannot express this compile at all on CUDA 13,
        # and the whole legacy-RRTMG lane failed closed at construction
        # (MEASURED on cupy 14.1.1 / CUDA 13.0.2 / sm_120: RawModule with
        # --ftz=false raises, RawModule without it compiles with
        # subnormals flushed, which is the wrong answer quietly).
        # compile_using_nvrtc + cuda.function.Module is the bypass
        # rrtmg_lw.py Section 10 and rrtmg_mcica.py already use: CuPy does
        # not inject -ftz on that path, so --ftz=false survives.  Do NOT
        # pass ``-arch`` here -- cupy appends its own
        # (cuda/compiler.py:_compile), and NVRTC 13 rejects that duplicate
        # too; the arch cupy derives is ``_get_arch()``, the same value an
        # explicit option would have carried.
        from woof import nvrtc_ptx_cache as _cc
        ptx, _mapping = _cc.compile_using_nvrtc(
            code, ("-std=c++17", "--ftz=false"), None, "rrtmg_sw.cu")
        self.module = cp.cuda.function.Module()
        self.module.load(ptx.encode() if isinstance(ptx, str) else ptx)
        from woof.certify.kernel_manifest import record_module
        record_module("woof.core.rrtmg_sw:rrtmg_sw", source=code,
                      options=("-std=c++17", "--ftz=false"),
                      module=self.module)
        self.tab_gpu = cp.asarray(packed)
        self.ngb_gpu = cp.asarray(np.asarray(tab.ngb, dtype=np.int32))
        self._laysolfr_spec_gpu = cp.asarray(
            np.asarray(self._laysolfr_band_spec(), dtype=np.int32))
        # These immutable arrays are shared across member streams. Publish
        # their readiness after all three uploads, before the cached engine
        # can be borrowed by another stream. Recording does not drain the GPU.
        self._tables_ready = cp.cuda.Event(disable_timing=True)
        self._tables_ready.record()
        self._table_streams = {}
        self.max_nlay = None  # retained API attribute: workspace has no fixed layer ceiling
        #: Per-chunk device workspace of the batched chain (Section 11):
        #: every slot is allocated once and reused across the chunks of a
        #: call, and only the slots the kernels read before writing are
        #: zeroed per chunk; the call itself hands the bytes back to the
        #: pool when it returns (release_scratch stays for callers).  ONE PER
        #: STREAM (see :attr:`_scratch`): the compiled module and the packed
        #: tables are read-only and shared by every stream on this card,
        #: the workspace is written by every call and is not.
        self._scratch_by_stream = {}
        #: nlayers -> columns per chunk, derived once per layer count
        #: from this device's resident threads and the VRAM free at that
        #: moment (see batch_column_chunk).
        self._chunk_by_nlayers = {}

    def _k(self, name):
        current = self.cp.cuda.get_current_stream()
        ptr = int(current.ptr)
        if ptr not in self._table_streams:
            current.wait_event(self._tables_ready)
            # Keep the stream alive so a destroyed stream's pointer cannot be
            # reused and mistaken for a queue that already waited. Each queue
            # pays one wait, before its first scalar or batched SW kernel.
            self._table_streams[ptr] = current
        return self.module.get_function(name)

    # ---- batched-chain workspace and chunk width ----------------------

    @property
    def _scratch(self):
        """The CURRENT stream's :class:`SWBatchScratch`.

        Two states stepping at the same time on one card (the slabs of a
        [devices] split, or a tiled run's buffers on their own streams) run
        this engine's batched chain concurrently.  One shared workspace was
        then written by both: MEASURED on the city suite, a slab lost bits
        in state/al and one threaded 2x2 run stopped on a non-finite
        rthratensw.  Keyed by the stream the call runs on, so a resident
        run (one stream) keeps exactly the one workspace it always had.
        """
        ptr = int(self.cp.cuda.get_current_stream().ptr)
        held = self._scratch_by_stream.get(ptr)
        if held is None:
            held = self._scratch_by_stream.setdefault(ptr, SWBatchScratch(self.cp))
        return held

    @property
    def scratch(self):
        """The batched chain's :class:`SWBatchScratch` (allocation and
        zeroing counters live on it)."""
        return self._scratch

    def release_scratch(self):
        """Hand every workspace slot back to the pool.  Callers invoke it
        after a radiation event so the transient returns to the driver
        at the model's next pool trim, exactly as the per-chunk
        allocations it replaces did."""
        self._scratch.release()

    def batch_column_chunk(self, nlayers):
        """Columns per chunk for ``nlayers`` on this device, derived once
        per layer count: the width that saturates the device's resident
        threads, narrowed only if that width's priced transient
        (:func:`sw_batched_vram_bytes`) would not fit in the VRAM free
        when first asked.  Chunk width is workspace shape only; results
        are bitwise identical at any width."""
        nlayers = _radiation_layer_count(nlayers)
        chunk = self._chunk_by_nlayers.get(nlayers)
        if chunk is None:
            chunk = sw_batch_column_chunk(
                nlayers, free_bytes=sw_batch_free_device_bytes(
                    self.cp, self._scratch))
            self._chunk_by_nlayers[nlayers] = chunk
        return chunk

    # ---- stage drivers (all arrays FP32; returns cupy arrays) ----

    def setcoef(self, nlayers, pavel, tavel, coldry, wkl):
        nlayers = _radiation_layer_count(nlayers)
        cp = self.cp
        f = lambda a: cp.asarray(np.asarray(a, np.float32))
        pavel, tavel, coldry = f(pavel[:nlayers]), f(tavel[:nlayers]), f(coldry[:nlayers])
        wkl = cp.asarray(np.asfortranarray(np.asarray(wkl[:, :nlayers], np.float32)))
        ints = {k: cp.zeros(nlayers, cp.int32) for k in
                ("jp", "jt", "jt1", "indself", "indfor", "tflag", "lflag")}
        reals = {k: cp.zeros(nlayers, cp.float32) for k in
                 ("colh2o", "colco2", "colo3", "coln2o", "colch4", "colo2",
                  "colmol", "co2mult", "selffac", "selffrac", "forfac",
                  "forfrac", "fac00", "fac01", "fac10", "fac11")}
        self._k("rsw_setcoef")(((nlayers + 63) // 64,), (64,), (
            np.int32(nlayers), self.tab_gpu, pavel, tavel, coldry, wkl,
            ints["jp"], ints["jt"], ints["jt1"], ints["indself"],
            ints["indfor"], ints["tflag"], ints["lflag"],
            reals["colh2o"], reals["colco2"], reals["colo3"], reals["coln2o"],
            reals["colch4"], reals["colo2"], reals["colmol"], reals["co2mult"],
            reals["selffac"], reals["selffrac"], reals["forfac"],
            reals["forfrac"], reals["fac00"], reals["fac01"], reals["fac10"],
            reals["fac11"]))
        out = dict(reals)
        out.update({k: ints[k] for k in ("jp", "jt", "jt1", "indself", "indfor")})
        out["laytrop"] = int(ints["tflag"].sum().get())
        out["laylow"] = int(ints["lflag"].sum().get())
        out["layswtch"] = 0
        return out

    def _laysolfr(self, laytrop, nlayers, jp_host):
        lay = np.zeros(14, dtype=np.int32)
        for i, band in enumerate(range(16, 30)):
            lr = self.tab.kg[band].get("layreffr")
            if lr is None:
                lay[i] = laytrop          # band 26
                continue
            lr = int(lr)
            if band in (16, 17, 27, 28, 29):
                lay[i] = _laysolfr_upper(laytrop, nlayers, jp_host, lr)
            else:
                lay[i] = _laysolfr_lower(laytrop, jp_host, lr)
        return lay

    def _laysolfr_band_spec(self):
        """(layreffr, is_upper, has_layreffr) for SW bands 16..29.

        Host-side table constants, read once per engine instance.
        """
        spec = getattr(self, "_laysolfr_spec", None)
        if spec is None:
            layreffr = np.zeros(NBNDSW, dtype=np.int32)
            is_upper = np.zeros(NBNDSW, dtype=bool)
            has_lr = np.zeros(NBNDSW, dtype=bool)
            for i, band in enumerate(range(16, 16 + NBNDSW)):
                value = self.tab.kg[band].get("layreffr")
                if value is None:            # band 26: laysolfr = laytrop
                    continue
                has_lr[i] = True
                layreffr[i] = int(value)
                is_upper[i] = band in (16, 17, 27, 28, 29)
            spec = (layreffr, is_upper, has_lr)
            self._laysolfr_spec = spec
        return spec

    def _laysolfr_batch_device(self, jp_d, laytrop_d, nlayers):
        """Batch laysolfr entirely on device -- exact device twin of the
        host scans :func:`_laysolfr_lower` / :func:`_laysolfr_upper`.

        Both scans are pure INTEGER comparisons over jp with no floating
        point anywhere, so the CUDA scan is bit-exact by
        construction rather than by tolerance.  Reproduced exactly:

        lower  laysolfr = laytrop; for lay = 1..laytrop, if
               jp[lay-1] < lr and jp[lay] >= lr then
               laysolfr = min(lay+1, laytrop).  The LAST satisfying lay
               wins (the scalar loop keeps overwriting).
        upper  laysolfr = nlayers; for lay = laytrop+1..nlayers, if
               jp[lay-2] < lr and jp[lay-1] >= lr then laysolfr = lay.
               Last wins.  At lay == 1 the scalar code indexes jp[-1],
               i.e. jp[nlayers-1]; the modulo below reproduces that
               Python wrap rather than silently diverging from it.

        jp[lay] at lay == nlayers is unreachable in the lower scan
        (it needs lay <= laytrop <= nlayers-1, and the scalar code would
        raise IndexError otherwise), so clamping that gather index is
        never observable.
        """
        cp = self.cp
        nc = int(jp_d.shape[0])
        result = cp.empty((nc, NBNDSW), dtype=cp.int32)
        self._k("rsw_laysolfr_b")(((nc * NBNDSW + 127) // 128,), (128,),
            (np.int32(nc), np.int32(nlayers), jp_d, laytrop_d,
             self._laysolfr_spec_gpu, result))
        return result

    def taumol(self, nlayers, sc):
        cp = self.cp
        taug = cp.zeros((nlayers, NGPTSW), cp.float32, order="F")
        taur = cp.zeros((nlayers, NGPTSW), cp.float32, order="F")
        sfluxzen = cp.zeros(NGPTSW, cp.float32)
        n = nlayers * NGPTSW
        args = (np.int32(nlayers), np.int32(sc["laytrop"]), self.tab_gpu,
                self.ngb_gpu, sc["jp"], sc["jt"], sc["jt1"],
                sc["colh2o"], sc["colco2"], sc["colch4"], sc["colo2"],
                sc["colo3"], sc["colmol"],
                sc["fac00"], sc["fac01"], sc["fac10"], sc["fac11"],
                sc["selffac"], sc["selffrac"], sc["indself"],
                sc["forfac"], sc["forfrac"], sc["indfor"], taug, taur)
        self._k("rsw_taumol")(((n + 127) // 128,), (128,), args)
        jp_host = sc["jp"].get()
        laysolfr = self._laysolfr(sc["laytrop"], nlayers, jp_host)
        self._k("rsw_sfluxzen")((1,), (128,), (
            np.int32(nlayers), np.int32(sc["laytrop"]), self.tab_gpu,
            self.ngb_gpu, cp.asarray(laysolfr),
            sc["jp"], sc["jt"], sc["jt1"],
            sc["colh2o"], sc["colco2"], sc["colch4"], sc["colo2"],
            sc["colo3"], sc["fac00"], sc["fac01"], sc["fac10"], sc["fac11"],
            sfluxzen))
        return sfluxzen, taug, taur

    def cldprmc(self, nlayers, inflag, iceflag, liqflag, cldfmc, ciwpmc,
                clwpmc, cswpmc, reicmc, relqmc, resnmc, taucmc, ssacmc,
                asmcmc, fsfcmc):
        cp = self.cp
        # cupy .copy() defaults to C order; the kernel indexes F order
        taormc = cp.array(taucmc, order="F", copy=True)
        err = cp.zeros(1, cp.int32)
        n = nlayers * NGPTSW
        self._k("rsw_cldprmc")(((n + 127) // 128,), (128,), (
            np.int32(nlayers), np.int32(inflag), np.int32(iceflag),
            np.int32(liqflag), self.tab_gpu, self.ngb_gpu,
            cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc,
            taormc, taucmc, ssacmc, asmcmc, fsfcmc, err))
        if int(err.get()[0]) != 0:
            raise ValueError(f"rsw_cldprmc error flag {int(err.get()[0])}")
        return taormc

    def spcvmc(self, nlayers, palbd, palbp, pcldfmc, ptaucmc, pasycmc,
               pomgcmc, ptaormc, ptaua, pasya, pomga, prmu0, adjflux, sc):
        cp = self.cp
        nlayers = _radiation_layer_count(nlayers)
        sfluxzen, taug, taur = self.taumol(nlayers, sc)
        nl1 = nlayers + 1
        g = lambda: cp.zeros((nl1, NGPTSW), cp.float32, order="F")
        zcd, zcu, zfd, zfu = g(), g(), g(), g()
        ztdbt_nodel, ztdbtc_nodel = g(), g()
        zincflx = cp.zeros(NGPTSW, cp.float32)
        # Per-thread workspace for the spcvmc pipeline (the arrays that
        # were thread-local until the local-frame audit; see the .cu).
        # Fully written before read in the same statement order as the
        # local-array version, so zeros vs garbage cannot matter.
        wk = cp.zeros((NGPTSW, SPCVMC_WK_ARRAYS * nl1), cp.float32)
        wkc = cp.zeros((NGPTSW, SPCVMC_WKC_ARRAYS * nl1), cp.uint8)
        self._k("rsw_spcvmc_gpt")((1,), (128,), (
            np.int32(nlayers), self.tab_gpu, self.ngb_gpu,
            palbd, palbp, pcldfmc, ptaucmc, pasycmc, pomgcmc, ptaormc,
            ptaua, pasya, pomga, np.float32(prmu0), adjflux, sfluxzen,
            taug, taur, zincflx, zcd, zcu, zfd, zfu,
            ztdbt_nodel, ztdbtc_nodel, wk, wkc))
        outs = {k: cp.zeros(nl1, cp.float32) for k in
                ("pbbfd", "pbbfu", "pbbcd", "pbbcu", "pbbfddir", "pbbcddir",
                 "puvfd", "puvcd", "puvfddir", "puvcddir",
                 "pnifd", "pnicd", "pnifddir", "pnicddir")}
        self._k("rsw_spc_accum")(((nl1 + 63) // 64,), (64,), (
            np.int32(nlayers), self.ngb_gpu, zincflx, zcd, zcu, zfd, zfu,
            ztdbt_nodel, ztdbtc_nodel,
            outs["pbbfd"], outs["pbbfu"], outs["pbbcd"], outs["pbbcu"],
            outs["pbbfddir"], outs["pbbcddir"],
            outs["puvfd"], outs["puvcd"], outs["puvfddir"], outs["puvcddir"],
            outs["pnifd"], outs["pnicd"], outs["pnifddir"], outs["pnicddir"]))
        outs.update(zincflx=zincflx, zcd=zcd, zcu=zcu, zfd=zfd, zfu=zfu,
                    zsflxzen=sfluxzen, ztaug=taug, ztaur=taur)
        return outs

    def rrtmg_sw(self, nlay, icld, play, plev, tlay, tlev, tsfc,
                 h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
                 asdir, asdif, aldir, aldif, coszen, adjes, dyofyr, scon,
                 inflgsw, iceflgsw, liqflgsw,
                 cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl,
                 ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl,
                 aer_opt=0, aerosol=None):
        """CUDA twin of :func:`rrtmg_sw` (zero-aerosol option-4 path).

        The campaign path always has zero aerosol arrays, so taua/ssaa/asma
        are constructed directly (0/1/0) as inatm does for iaer=10 with
        zero inputs.  Returns numpy arrays (downloaded once at the end).
        """
        cp = self.cp
        if aer_opt in (1,2,3):
            if aerosol is None or len(aerosol)!=3:
                raise ValueError(f"aer_opt={aer_opt} needs caller-supplied aerosol optics; missing arrays would run clean shortwave")
            columns=lambda a:cp.asarray(a,dtype=cp.float32).reshape(1,-1)
            scalar=lambda a:np.asarray([a],np.float32)
            mc=lambda a:cp.asarray(a,dtype=cp.float32)[:,None,:]
            tau,ssa,asy=aerosol
            optics=tuple(cp.ascontiguousarray(cp.asarray(a,dtype=cp.float32).T[None,:,:]) for a in (tau,asy,ssa))
            result=self.rrtmg_sw_batched_device(
                1,nlay,icld,*[columns(a) for a in (play,plev,tlay,tlev)],scalar(tsfc),
                *[columns(a) for a in (h2ovmr,o3vmr,co2vmr,ch4vmr,n2ovmr,o2vmr)],
                *[scalar(a) for a in (asdir,asdif,aldir,aldif,coszen,adjes)],dyofyr,scalar(scon),
                inflgsw,iceflgsw,liqflgsw,
                *[mc(a) for a in (cldfmcl,taucmcl,ssacmcl,asmcmcl,fsfcmcl,ciwpmcl,clwpmcl,cswpmcl)],
                *[columns(a) for a in (reicmcl,relqmcl,resnmcl)],aer_opt=aer_opt,aerosol=optics)
            return {key:value[0] for key,value in sw_batched_to_host(result).items()}
        if aer_opt != 0:
            raise NotImplementedError(f"aer_opt={aer_opt!r} is not a WRF aerosol option this twin carries; clean air would replace its optics")
        if aerosol is not None:
            raise ValueError("caller-supplied optics with aer_opt=0 would be discarded")
        if int(dyofyr) > 0:
            earth_sun(dyofyr)
        nlayers = _radiation_layer_count(nlay)
        f = lambda a: cp.asarray(np.asarray(a, np.float32))

        # inatm: host scalar prep (adjflux/solvar/tbound), device coldry/wkl
        adjflx = F(adjes)
        solvar = np.zeros(NBNDSW, dtype=np.float32)
        adjflux = np.zeros(NBNDSW, dtype=np.float32)
        for ib in range(NBNDSW):
            solvar[ib] = F(F(scon) / RRSW_SCON)
            adjflux[ib] = F(adjflx * solvar[ib])

        plev_g = f(plev)
        pdp = cp.zeros(nlayers, cp.float32)
        coldry = cp.zeros(nlayers, cp.float32)
        wkl = cp.zeros((MXMOL, nlayers), cp.float32, order="F")
        self._k("rsw_inatm_layers")(((nlayers + 63) // 64,), (64,), (
            np.int32(nlayers), np.float32(self.tab.grav),
            np.float32(self.tab.avogad), plev_g,
            f(h2ovmr[:nlayers]), f(co2vmr[:nlayers]), f(o3vmr[:nlayers]),
            f(n2ovmr[:nlayers]), f(ch4vmr[:nlayers]), f(o2vmr[:nlayers]),
            pdp, coldry, wkl))

        cldfmc = cp.asfortranarray(f(cldfmcl[:, :nlayers]))
        taucmc = cp.asfortranarray(f(taucmcl[:, :nlayers]))
        ssacmc = cp.asfortranarray(f(ssacmcl[:, :nlayers]))
        asmcmc = cp.asfortranarray(f(asmcmcl[:, :nlayers]))
        fsfcmc = cp.asfortranarray(f(fsfcmcl[:, :nlayers]))
        ciwpmc = cp.asfortranarray(f(ciwpmcl[:, :nlayers]))
        clwpmc = cp.asfortranarray(f(clwpmcl[:, :nlayers]))
        if int(iceflgsw) == 5:
            cswpmc = cp.asfortranarray(f(cswpmcl[:, :nlayers]))
            resnmc = f(resnmcl[:nlayers])
        else:
            cswpmc = cp.zeros((NGPTSW, nlayers), cp.float32, order="F")
            resnmc = cp.zeros(nlayers, cp.float32)
        reicmc = f(reicmcl[:nlayers])
        relqmc = f(relqmcl[:nlayers])
        if icld < 1:
            raise NotImplementedError(
                "icld = 0 (clear-only) never occurs in the campaign "
                "(cldovrlp = 2); fails closed")

        taormc = self.cldprmc(nlayers, int(inflgsw), int(iceflgsw),
                              int(liqflgsw), cldfmc, ciwpmc, clwpmc, cswpmc,
                              reicmc, relqmc, resnmc, taucmc, ssacmc,
                              asmcmc, fsfcmc)

        sc = self.setcoef(nlayers, np.asarray(play, np.float32),
                          np.asarray(tlay, np.float32),
                          coldry.get(), wkl.get())
        # keep device copies for taumol
        sc_dev = dict(sc)

        cossza = F(coszen)
        if cossza <= F(1.0e-10):
            cossza = F(1.0e-10)

        albdir = np.zeros(NBNDSW, dtype=np.float32)
        albdif = np.zeros(NBNDSW, dtype=np.float32)
        albdir[:9] = F(aldir); albdif[:9] = F(aldif)
        albdir[13] = F(aldir); albdif[13] = F(aldif)
        albdir[9:13] = F(asdir); albdif[9:13] = F(asdif)

        def t2f(src):
            # elementwise transpose copy; cp.asfortranarray(x.T) would go
            # through cuBLAS, which this environment cannot load
            out = cp.zeros((src.shape[1], src.shape[0]), cp.float32,
                           order="F")
            out[...] = src.T
            return out

        zcldfmc = t2f(cldfmc)
        ztaucmc = t2f(taucmc)
        ztaormc = t2f(taormc)
        zasycmc = t2f(asmcmc)
        zomgcmc = t2f(ssacmc)
        ztaua = cp.zeros((nlayers, NBNDSW), cp.float32, order="F")
        zasya = cp.zeros((nlayers, NBNDSW), cp.float32, order="F")
        zomga = cp.ones((nlayers, NBNDSW), cp.float32, order="F")

        out = self.spcvmc(nlayers, cp.asarray(albdif), cp.asarray(albdir),
                          zcldfmc, ztaucmc, zasycmc, zomgcmc, ztaormc,
                          ztaua, zasya, zomga, cossza, cp.asarray(adjflux),
                          sc_dev)

        swhr = cp.zeros(nlayers, cp.float32)
        swhrc = cp.zeros(nlayers, cp.float32)
        self._k("rsw_post")(((nlayers + 63) // 64,), (64,), (
            np.int32(nlayers), np.float32(self.tab.heatfac), pdp,
            out["pbbfu"], out["pbbfd"], out["pbbcu"], out["pbbcd"],
            swhr, swhrc))

        nl1 = nlayers + 1
        swuflx = out["pbbfu"].get()
        swdflx = out["pbbfd"].get()
        swuflxc = out["pbbcu"].get()
        swdflxc = out["pbbcd"].get()
        dirdflux = out["pbbfddir"].get()
        puvfd = out["puvfd"].get(); puvfddir = out["puvfddir"].get()
        pnifd = out["pnifd"].get(); pnifddir = out["pnifddir"].get()
        # single-op FP32 subtractions (bitwise-defined) on host
        swdkdif = np.zeros(nl1, np.float32)
        sibvisdif = np.zeros(nl1, np.float32)
        sibnirdif = np.zeros(nl1, np.float32)
        for i in range(nl1):
            swdkdif[i] = F(swdflx[i] - dirdflux[i])
            sibvisdif[i] = F(puvfd[i] - puvfddir[i])
            sibnirdif[i] = F(pnifd[i] - pnifddir[i])

        return dict(swuflx=swuflx, swdflx=swdflx, swhr=swhr.get(),
                    swuflxc=swuflxc, swdflxc=swdflxc, swhrc=swhrc.get(),
                    swuflxcln=np.zeros(nl1, np.float32),
                    swdflxcln=np.zeros(nl1, np.float32),
                    sibvisdir=puvfddir, sibvisdif=sibvisdif,
                    sibnirdir=pnifddir, sibnirdif=sibnirdif,
                    swdkdir=dirdflux, swdkdif=swdkdif,
                    swdkdirc=out["pbbcddir"].get())

    # ---- SECTION 11 methods: batched multi-column path ------------------

    def local_frame_bytes(self):
        """CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES for every kernel in the SW
        translation unit (single-column entries and their ``_b`` twins).

        A per-thread local frame of F bytes reserves roughly
        F x 1536 x 170 bytes machine-wide on the RTX 5090 at first launch
        (the KF_KMAX lesson), so frames are audited and bounded by
        tests/test_rrtmg_sw_cuda.py.  Module.get_function objects expose
        no ``.attributes`` property; the driver query is what works.
        """
        from cupy.cuda import driver
        out = {}
        for name in SW_GPU_KERNEL_NAMES:
            fn = self._k(name)
            # RawModule.get_function returns a RawKernel wrapper; the
            # CUfunction handle lives on its .kernel.
            ptr = getattr(fn, "kernel", fn).ptr
            out[name] = int(driver.funcGetAttribute(
                driver.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, ptr))
        return out

    def rrtmg_sw_batched_device(self, ncol, nlay, icld, play, plev, tlay,
                                tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr,
                                n2ovmr, o2vmr, asdir, asdif, aldir, aldif,
                                coszen, adjes, dyofyr, scon,
                                inflgsw, iceflgsw, liqflgsw,
                                cldfmcl, taucmcl, ssacmcl, asmcmcl,
                                fsfcmcl, ciwpmcl, clwpmcl, cswpmcl,
                                reicmcl, relqmcl, resnmcl, aer_opt=0,
                                column_chunk=None, _stage_probe=None,
                                mcica_layout="gpoint", aerosol=None):
        """Batched full SW chain, device-resident: the ``_b`` re-indexing
        twins of Section 10's kernels, same compile path, same per-thread
        statement order -- the only changes are grid sizes (ncol > 1) and
        host plumbing.

        DAY COLUMNS ONLY: every ``coszen`` must be > 0 (the WRF driver's
        night gate skips the SW call entirely; night columns have no
        defined SW output).  A batch containing any coszen <= 0 raises
        ValueError (fails closed) -- callers filter first.

        AEROSOL.  aer_opt = 0 builds WRF's zero-aerosol optics (0/1/0)
        internally, exactly as before.  aer_opt = 3 (the Thompson
        aerosol branch of the operational HRRR fork) takes the per-band
        optics the caller formed, ``aerosol = (ztaua, zasya, zomga)``,
        three float32 device arrays (ncol, NBNDSW, nlayers) in
        rsw_spcvmc_gpt_b's per-column band-major layout
        (woof.core.rrtmg_aerosol_optics builds them); RRTMG_SWRAD runs
        that branch with iaer = 10 and hands them to spcvmc unchanged
        (module_ra_rrtmg_sw.F:9183, :9329-9331).  aer_opt = 1 stays
        refused (their aerosol inputs are not transcribed; the optics
        would be silently discarded, 2026-07-27 SW transcription audit,
        item 1).
        Caller-supplied optics reach the raw asymmetry-combination
        division of rsw_spcvmc_gpt_b, whose operands the kernel header's
        invariant requires to be zero or normal float32;
        woof.core.rrtmg_aerosol_optics states and tests that range for
        the aer_opt = 3 producer.

        Layout (see the Section 11 header below): per-column arrays carry
        a leading ncol axis; McICA arrays default to (NGPTSW, ncol, nlay).
        ``mcica_layout="column"`` accepts (ncol, nlay, NGPTSW) and may
        read and write caller slabs through views, matching intent(inout).
        Callers must not reuse these slabs after this call.  The arrays
        tsfc/coszen/adjes/scon and the albedo splits asdir/asdif/aldir/
        aldif are (ncol,); icld/inflgsw/iceflgsw/liqflgsw/dyofyr are
        python ints SHARED by the batch (callers with mixed flags split
        the batch by flag tuple).  ``tsfc`` and ``tlev`` are accepted for
        signature parity with the per-column entry and are unused by the
        SW compute chain, exactly as there.  Inputs numpy or cupy;
        float32 is the contract dtype.

        Returns a dict of device (cupy) arrays; fetch host copies (and
        the three derived ``*dif`` outputs) via :func:`sw_batched_to_host`.
        ``column_chunk`` bounds the transient VRAM of the internal
        pipeline (:func:`sw_batched_vram_bytes` prices it). The abort flag
        is read once after all chunks. The device
        laysolfr scan matches the same host-side integer scans the
        per-column driver runs (data movement, no FP).  ``_stage_probe``
        is test instrumentation called at the allocation high-water
        stages; it must not affect results.
        """
        if mcica_layout not in ("gpoint", "column"):
            raise ValueError("mcica_layout must be gpoint or column")
        cp = self.cp
        if aer_opt in (1, 2, 3):
            if aerosol is None or len(aerosol) != 3:
                raise ValueError(
                    f"aer_opt={aer_opt} needs the caller's per-band aerosol optics "
                    "aerosol=(ztaua, zasya, zomga); without them the "
                    f"batch would run zero aerosol under aer_opt={aer_opt}")
        elif aer_opt != 0:
            # No other WRF aer_opt exists: running zero aerosol under an
            # unknown value would be silently wrong.
            raise NotImplementedError(
                f"aer_opt={aer_opt!r}: the CUDA SW composition implements "
                "aer_opt 0 (zero aerosol), 1, 2 and 3 (caller-supplied per-band "
                "optics); this option's aerosol inputs are not "
                "transcribed and would be silently discarded -- fails "
                "closed")
        elif aerosol is not None:
            raise ValueError("aerosol optics were supplied with aer_opt=0, "
                             "which runs WRF's zero aerosol")
        if icld < 1:
            raise NotImplementedError(
                "icld = 0 (clear-only) never occurs in the campaign "
                "(cldovrlp = 2); fails closed")
        if int(dyofyr) > 0:
            earth_sun(dyofyr)   # raises: unreachable via WRF option 4
        nlayers = _radiation_layer_count(nlay)
        ncol = int(ncol)
        nl1 = nlayers + 1
        if aerosol is not None:
            for name, a in zip(("ztaua", "zasya", "zomga"), aerosol):
                if (not isinstance(a, cp.ndarray) or a.dtype != cp.float32
                        or a.shape != (ncol, NBNDSW, nlayers)
                        or not a.flags.c_contiguous):
                    raise ValueError(
                        f"aer_opt={aer_opt} {name} must be a C-contiguous float32 "
                        f"device array {(ncol, NBNDSW, nlayers)}, got "
                        f"{getattr(a, 'dtype', None)} "
                        f"{getattr(a, 'shape', None)}")
        if column_chunk:
            chunk = int(column_chunk)
        else:
            chunk = self.batch_column_chunk(nlayers)
        if chunk < 1:
            raise ValueError("column_chunk must be >= 1")
        i32, f32 = np.int32, np.float32
        scratch = self._scratch

        def hostf(a):
            h = cp.asnumpy(a) if isinstance(a, cp.ndarray) else a
            return np.asarray(h, f32).reshape(ncol)

        def _day_verdict(night):
            if bool(night):
                raise ValueError(
                    "batched SW is day-columns-only (every coszen must be "
                    "> 0); the WRF option-4 driver's night gate skips the SW "
                    "call entirely -- filter night columns before batching")

        # ---- per-column scalar prep (same single-op FP32 chains as the
        # per-column driver, elementwise).  Device inputs stay on the
        # device: a where, one IEEE division and one product, which CUDA
        # rounds exactly as NumPy float32 does, and copies; reading them
        # back each chunk drained the card before every SW chunk (A3).
        device = all(isinstance(a, cp.ndarray) and a.dtype == cp.float32
                     for a in (coszen, aldir, aldif, asdir, asdif))
        xp = cp if device else np
        if device:
            cz = cp.ascontiguousarray(coszen.reshape(ncol))
            # ---- day-columns-only contract (fail closed) ----------------
            from woof.core.deferred_device_checks import check
            check(cp.any(~(cz > F(0.0))), _day_verdict)
            dev = lambda a: (cp.ascontiguousarray(a.reshape(ncol))
                             if isinstance(a, cp.ndarray) and a.dtype == cp.float32
                             else cp.asarray(hostf(a)))
        else:
            cz = hostf(coszen)
            _day_verdict(not bool(np.all(cz > F(0.0))))
            dev = hostf
        cossza_h = xp.where(cz <= F(1.0e-10), F(1.0e-10), cz)
        adjflx_h = dev(adjes)
        solvar_h = dev(scon) / RRSW_SCON          # F(F(scon) / RRSW_SCON)
        adjb_h = adjflx_h * solvar_h              # F(adjflx * solvar[ib])
        adjflux_h = xp.repeat(adjb_h[:, None], NBNDSW, axis=1)
        aldir_h, aldif_h = dev(aldir), dev(aldif)
        asdir_h, asdif_h = dev(asdir), dev(asdif)
        albdir_h = xp.zeros((ncol, NBNDSW), f32)
        albdif_h = xp.zeros((ncol, NBNDSW), f32)
        albdir_h[:, 0:9] = aldir_h[:, None]
        albdif_h[:, 0:9] = aldif_h[:, None]
        albdir_h[:, NBNDSW - 1] = aldir_h
        albdif_h[:, NBNDSW - 1] = aldif_h
        albdir_h[:, 9:13] = asdir_h[:, None]
        albdif_h[:, 9:13] = asdif_h[:, None]

        # ---- batch-level output slabs (alive for the whole call, and
        # returned: never workspace).  Every row of every slab is
        # assigned by the chunk loop below, so they are not zeroed. ------
        O = {k: cp.empty((ncol, nl1), dtype=cp.float32)
             for k in ("swuflx", "swdflx", "swuflxc", "swdflxc",
                       "swdkdir", "swdkdirc", "sibvisdir", "sibnirdir",
                       "puvfd", "pnifd")}
        swhr = cp.empty((ncol, nlayers), dtype=cp.float32)
        swhrc = cp.empty((ncol, nlayers), dtype=cp.float32)

        iceflg = int(iceflgsw)
        # laysolfr is computed on device from the resident jp/laytrop
        # (see _laysolfr_batch_device); the old host memo was rebuilt on
        # every call, so it never survived a step and is gone.
        #
        # Every per-chunk device array below that a kernel fully writes
        # before anything reads it is a ``scratch.take`` slot: allocated
        # once (sized to the widest chunk seen) and reused, never zeroed.
        # The read-before-write slots are ``scratch.zeros`` and are
        # zeroed per chunk; constant kernel inputs are ``scratch.constant``
        # and are filled once.  SW_SCRATCH_SLOTS lists every slot with
        # the kernel fact that puts it in its class.
        # Shared atomic abort word is cleared once for the entire call.
        err_d = scratch.zeros("err", (1,), i32)
        for c0 in range(0, ncol, chunk):
            c1 = min(c0 + chunk, ncol)
            nc = c1 - c0
            rows = slice(c0, c1)

            # ---- uploads + inatm ------------------------------------
            plev_d = _sw_dev_chunk(cp, plev, rows, (slice(0, nl1),))
            play_d = _sw_dev_chunk(cp, play, rows, (slice(0, nlayers),))
            tlay_d = _sw_dev_chunk(cp, tlay, rows, (slice(0, nlayers),))
            vmr_d = [_sw_dev_chunk(cp, a, rows, (slice(0, nlayers),))
                     for a in (h2ovmr, co2vmr, o3vmr, n2ovmr, ch4vmr,
                               o2vmr)]
            pdp_d = scratch.take("pdp", (nc, nlayers), f32)
            coldry_d = scratch.take("coldry", (nc, nlayers), f32)
            # rsw_inatm_layers_b writes species 0-3, 5, 6 and then scales
            # all RSW_NMOL entries in place: species 4 is read unwritten
            # and must be zero (as the Fortran inatm zeroes wkl first).
            wkl_d = scratch.zeros("wkl", (nc, nlayers, MXMOL), f32)
            total = nc * nlayers
            self._k("rsw_inatm_layers_b")(
                ((total + 63) // 64,), (64,),
                (i32(nc), i32(nlayers), f32(self.tab.grav),
                 f32(self.tab.avogad), plev_d, *vmr_d,
                 pdp_d, coldry_d, wkl_d))
            del plev_d, vmr_d

            # ---- McICA uploads + cldprmc ----------------------------
            cldfmc_d = _sw_dev_mcica(cp, cldfmcl, c0, c1, nlayers, mcica_layout)
            taucmc_d = _sw_dev_mcica(cp, taucmcl, c0, c1, nlayers, mcica_layout)
            ssacmc_d = _sw_dev_mcica(cp, ssacmcl, c0, c1, nlayers, mcica_layout)
            asmcmc_d = _sw_dev_mcica(cp, asmcmcl, c0, c1, nlayers, mcica_layout)
            fsfcmc_d = _sw_dev_mcica(cp, fsfcmcl, c0, c1, nlayers, mcica_layout)
            ciwpmc_d = _sw_dev_mcica(cp, ciwpmcl, c0, c1, nlayers, mcica_layout)
            clwpmc_d = _sw_dev_mcica(cp, clwpmcl, c0, c1, nlayers, mcica_layout)
            reicmc_d = _sw_dev_chunk(cp, reicmcl, rows,
                                     (slice(0, nlayers),))
            relqmc_d = _sw_dev_chunk(cp, relqmcl, rows,
                                     (slice(0, nlayers),))
            if iceflg == 5:
                cswpmc_d = _sw_dev_mcica(cp, cswpmcl, c0, c1, nlayers, mcica_layout)
                resnmc_d = _sw_dev_chunk(cp, resnmcl, rows,
                                         (slice(0, nlayers),))
            else:
                # rsw_cldprmc_b takes both as const inputs (never
                # written), so the zero snow path is a filled-once
                # constant, not a per-chunk memset.
                cswpmc_d = scratch.constant(
                    "cswpmc0", (nc, nlayers, NGPTSW), f32, 0.0)
                resnmc_d = scratch.constant(
                    "resnmc0", (nc, nlayers), f32, 0.0)
            taormc_d = taucmc_d.copy()
            # The abort flag is written by the kernel only on failure
            # and checked once after all chunks.
            total = nc * nlayers * NGPTSW
            self._k("rsw_cldprmc_b")(
                ((total + 127) // 128,), (128,),
                (i32(nc), i32(nlayers), i32(inflgsw), i32(iceflg),
                 i32(liqflgsw), self.tab_gpu, self.ngb_gpu,
                 cldfmc_d, ciwpmc_d, clwpmc_d, cswpmc_d,
                 reicmc_d, relqmc_d, resnmc_d,
                 taormc_d, taucmc_d, ssacmc_d, asmcmc_d, fsfcmc_d,
                 err_d))
            if _stage_probe is not None:
                _stage_probe("upload")
            del (ciwpmc_d, clwpmc_d, cswpmc_d, fsfcmc_d, reicmc_d,
                 relqmc_d, resnmc_d)

            # ---- setcoef (every output written per (column, layer)) --
            ints_d = {k: scratch.take(k, (nc, nlayers), i32)
                      for k in SETCOEF_INT_SLOTS}
            reals_d = {k: scratch.take(k, (nc, nlayers), f32)
                       for k in SETCOEF_REAL_SLOTS}
            total = nc * nlayers
            self._k("rsw_setcoef_b")(
                ((total + 63) // 64,), (64,),
                (i32(nc), i32(nlayers), self.tab_gpu,
                 play_d, tlay_d, coldry_d, wkl_d,
                 ints_d["jp"], ints_d["jt"], ints_d["jt1"],
                 ints_d["indself"], ints_d["indfor"],
                 ints_d["tflag"], ints_d["lflag"],
                 reals_d["colh2o"], reals_d["colco2"], reals_d["colo3"],
                 reals_d["coln2o"], reals_d["colch4"], reals_d["colo2"],
                 reals_d["colmol"], reals_d["co2mult"],
                 reals_d["selffac"], reals_d["selffrac"],
                 reals_d["forfac"], reals_d["forfrac"],
                 reals_d["fac00"], reals_d["fac01"], reals_d["fac10"],
                 reals_d["fac11"]))
            # laytrop per column: integer sum of the tropopause flags
            # (exact).  jp and laytrop stay resident: laysolfr is the
            # same integer scan, run on device (S3a of the perf wave).
            laytrop_d = ints_d["tflag"].sum(axis=1, dtype=cp.int32)
            del play_d, tlay_d, coldry_d, wkl_d

            laysolfr_d = self._laysolfr_batch_device(
                ints_d["jp"], laytrop_d, nlayers)

            # ---- taumol + sfluxzen (one write per thread, every
            # (column, g-point, layer) and (column, g-point) covered) ---
            taug_d = scratch.take("taug", (nc, nlayers, NGPTSW), f32)
            taur_d = scratch.take("taur", (nc, nlayers, NGPTSW), f32)
            total = nc * nlayers * NGPTSW
            self._k("rsw_taumol_b")(
                ((total + 127) // 128,), (128,),
                (i32(nc), i32(nlayers), laytrop_d, self.tab_gpu,
                 self.ngb_gpu, ints_d["jp"], ints_d["jt"], ints_d["jt1"],
                 reals_d["colh2o"], reals_d["colco2"], reals_d["colch4"],
                 reals_d["colo2"], reals_d["colo3"], reals_d["colmol"],
                 reals_d["fac00"], reals_d["fac01"], reals_d["fac10"],
                 reals_d["fac11"], reals_d["selffac"],
                 reals_d["selffrac"], ints_d["indself"],
                 reals_d["forfac"], reals_d["forfrac"],
                 ints_d["indfor"], taug_d, taur_d))
            sflux_d = scratch.take("sflux", (nc, NGPTSW), f32)
            total = nc * NGPTSW
            self._k("rsw_sfluxzen_b")(
                ((total + 127) // 128,), (128,),
                (i32(nc), i32(nlayers), laytrop_d, self.tab_gpu,
                 self.ngb_gpu, laysolfr_d,
                 ints_d["jp"], ints_d["jt"], ints_d["jt1"],
                 reals_d["colh2o"], reals_d["colco2"], reals_d["colch4"],
                 reals_d["colo2"], reals_d["colo3"],
                 reals_d["fac00"], reals_d["fac01"], reals_d["fac10"],
                 reals_d["fac11"], sflux_d))
            if _stage_probe is not None:
                _stage_probe("coef")
            del ints_d, reals_d, laytrop_d, laysolfr_d

            # Cloud optics already has the coalesced (column, layer, g-point) layout.
            zcldfmc_d, ztaucmc_d, ztaormc_d = cldfmc_d, taucmc_d, taormc_d
            zasycmc_d, zomgcmc_d = asmcmc_d, ssacmc_d
            del cldfmc_d, taucmc_d, taormc_d, asmcmc_d, ssacmc_d
            if aerosol is None:
                # Zero-aerosol optics (aer_opt = 0): const inputs of
                # rsw_spcvmc_gpt_b, filled once.
                ztaua_d = scratch.constant(
                    "ztaua", (nc, NBNDSW, nlayers), f32, 0.0)
                zasya_d = scratch.constant(
                    "zasya", (nc, NBNDSW, nlayers), f32, 0.0)
                zomga_d = scratch.constant(
                    "zomga", (nc, NBNDSW, nlayers), f32, 1.0)
            else:
                # aer_opt = 3: the caller's optics for these columns
                # (contiguous row views, the same layout and dtype).
                ztaua_d = aerosol[0][rows]
                zasya_d = aerosol[1][rows]
                zomga_d = aerosol[2][rows]

            albdif_d = cp.ascontiguousarray(cp.asarray(albdif_h[rows]))
            albdir_d = cp.ascontiguousarray(cp.asarray(albdir_h[rows]))
            adjflux_d = cp.ascontiguousarray(cp.asarray(adjflux_h[rows]))
            cossza_d = cp.ascontiguousarray(cp.asarray(cossza_h[rows]))

            # ---- spcvmc ---------------------------------------------
            # wk: each thread owns RSW_SPCVMC_WK x n1 entries, plus
            # a zero-byte wkc ABI placeholder, interleaved across the nthr
            # threads of the launch (entry j of array a for thread t at
            # (a * n1 + j) * nthr + t, kernels/rrtmg_sw.cu RswStrided, so
            # a warp's accesses coalesce; the constants price its geometry),
            # and writes every entry it later reads (rsw_spcvmc_body,
            # fused layer pass, rsw_vrtqdr), so workspace enters each chunk
            # as it left the last.  The six outputs and zincflx are written for
            # every (column, g-point[, level]) before rsw_spc_accum_b
            # reads them.
            nthr = nc * NGPTSW
            wk_d = scratch.take("wk", (nthr, SPCVMC_WK_ARRAYS * nl1), f32)
            wkc_d = scratch.take("wkc", (nthr, SPCVMC_WKC_ARRAYS * nl1),
                                 np.uint8)
            zincflx_d = scratch.take("zincflx", (nc, NGPTSW), f32)
            zouts = [scratch.take(slot, (nc, nl1, NGPTSW), f32)
                     for slot in SPCVMC_OUT_SLOTS]
            zcd_d, zcu_d, zfd_d, zfu_d, ztn_d, ztcn_d = zouts
            self._k("rsw_spcvmc_gpt_b")(
                ((nthr + 127) // 128,), (128,),
                (i32(nc), i32(nlayers), self.tab_gpu, self.ngb_gpu,
                 albdif_d, albdir_d, zcldfmc_d, ztaucmc_d, zasycmc_d,
                 zomgcmc_d, ztaormc_d, ztaua_d, zasya_d, zomga_d,
                 cossza_d, adjflux_d, sflux_d, taug_d, taur_d,
                 zincflx_d, zcd_d, zcu_d, zfd_d, zfu_d, ztn_d, ztcn_d,
                 wk_d, wkc_d))
            if _stage_probe is not None:
                _stage_probe("spcvmc")
            del (wk_d, wkc_d, zcldfmc_d, ztaucmc_d, ztaormc_d,
                 zasycmc_d, zomgcmc_d, ztaua_d, zasya_d, zomga_d,
                 taug_d, taur_d, sflux_d, albdif_d, albdir_d,
                 adjflux_d, cossza_d, zouts)

            # ---- band accumulation + heating rates (rsw_spc_accum_b
            # sums in registers and stores once per (column, level);
            # rsw_post_b writes every (column, layer)) -----------------
            acc = {k: scratch.take(k, (nc, nl1), f32)
                   for k in SPC_ACCUM_SLOTS}
            total = nc * nl1
            self._k("rsw_spc_accum_b")(
                (nc, (nl1 + 31) // 32), (32,),
                (i32(nc), i32(nlayers), self.ngb_gpu, zincflx_d,
                 zcd_d, zcu_d, zfd_d, zfu_d, ztn_d, ztcn_d,
                 acc["pbbfd"], acc["pbbfu"], acc["pbbcd"], acc["pbbcu"],
                 acc["pbbfddir"], acc["pbbcddir"],
                 acc["puvfd"], acc["puvcd"], acc["puvfddir"],
                 acc["puvcddir"], acc["pnifd"], acc["pnicd"],
                 acc["pnifddir"], acc["pnicddir"]),
                shared_mem=6 * 8 * 32 * 4)
            del zincflx_d, zcd_d, zcu_d, zfd_d, zfu_d, ztn_d, ztcn_d

            swhr_d = scratch.take("swhr", (nc, nlayers), f32)
            swhrc_d = scratch.take("swhrc", (nc, nlayers), f32)
            total = nc * nlayers
            self._k("rsw_post_b")(
                ((total + 63) // 64,), (64,),
                (i32(nc), i32(nlayers), f32(self.tab.heatfac), pdp_d,
                 acc["pbbfu"], acc["pbbfd"], acc["pbbcu"], acc["pbbcd"],
                 swhr_d, swhrc_d))

            O["swuflx"][rows] = acc["pbbfu"]
            O["swdflx"][rows] = acc["pbbfd"]
            O["swuflxc"][rows] = acc["pbbcu"]
            O["swdflxc"][rows] = acc["pbbcd"]
            O["swdkdir"][rows] = acc["pbbfddir"]
            O["swdkdirc"][rows] = acc["pbbcddir"]
            O["sibvisdir"][rows] = acc["puvfddir"]
            O["puvfd"][rows] = acc["puvfd"]
            O["sibnirdir"][rows] = acc["pnifddir"]
            O["pnifd"][rows] = acc["pnifd"]
            swhr[rows] = swhr_d
            swhrc[rows] = swhrc_d
            # Drop the chunk's views before the next iteration: the
            # scratch keeps the buffers (that is the point), the uploads
            # return to the pool (keeps sw_batched_vram_bytes an upper
            # bound; data movement only).
            del acc, pdp_d, swhr_d, swhrc_d

        def _cldprmc_verdict(err):
            err = int(err[0])
            if err:
                raise ValueError(f"rsw_cldprmc device abort, code {err} "
                                 "(mirrors the Fortran STOPs)")

        # Read now, or once per radiation call
        # (woof.core.deferred_device_checks).  ``err`` is reused scratch
        # the next call zeroes, so the deferred read takes its own copy.
        from woof.core.deferred_device_checks import check
        check(err_d.copy(), _cldprmc_verdict)
        O["swhr"] = swhr
        O["swhrc"] = swhrc
        # The clean-sky-no-aerosol pair IS zero by the aer_opt = 0
        # contract; under aer_opt = 3 it stays zero because the operational
        # HRRR fork's rrtmg_sw has no clean-sky pass at all (no *cln
        # output in its module_ra_rrtmg_sw.F).  These two are the only
        # zeroed returned slabs.
        O["swuflxcln"] = cp.zeros((ncol, nl1), dtype=cp.float32)
        O["swdflxcln"] = cp.zeros((ncol, nl1), dtype=cp.float32)
        # The workspace served every chunk of this call; hand it back now
        # (memory census 2026-10-14, item 8).  A caller that never invoked
        # release_scratch (the hex MPAS RRTMG wrapper) held 1.6 GiB of it
        # through every non-radiation step at 120 km.  Releasing drops the
        # Python references only: the kernels queued on this stream keep
        # their blocks until they finish (CuPy's pool frees are ordered on
        # the allocating stream), every ``take`` slot is written before it
        # is read (SW_TAKE_SLOTS) and the constants are refilled with the
        # same words, so the next call computes the same bits.
        scratch.release()
        return O

    def rrtmg_sw_batched(self, *args, **kwargs):
        """rrtmg_sw_batched_device + host fetch: batched twin of
        :meth:`rrtmg_sw` (leading-ncol argument set, day columns only,
        numpy outputs with exactly the per-column key set)."""
        return sw_batched_to_host(
            self.rrtmg_sw_batched_device(*args, **kwargs))


# ===========================================================================
# SECTION 11: batched multi-column CUDA path (host plumbing)
#
# Section 10's kernels were written SINGLE-COLUMN (no ncol axis anywhere,
# grid (1,) launches) -- unlike the LW unit, whose kernels were ncol-major
# from the start.  Batching therefore adds rsw_*_b re-indexing twins in
# the same translation unit: each factors the original kernel into a
# __device__ body (arithmetic verbatim) called by both the original
# wrapper and a batched wrapper that maps thread -> (column, work item)
# and offsets array pointers to the column base.  No FP statement moved;
# per-thread arithmetic depends only on its own column's data (there are
# no cross-column reductions in the SW chain), so per-column results are
# bitwise identical at any batch width and chunk size -- proved by
# tests/test_rrtmg_sw_cuda.py over the full fixture decks.
#
# Batched input layout (= the per-column rrtmg_sw argument set with a
# leading ncol axis): play (ncol, nlay), plev (ncol, nlay+1), tlay/tlev
# likewise, gas vmr arrays (ncol, nlay), McICA arrays (NGPTSW, ncol,
# nlay), reicmcl/relqmcl/resnmcl (ncol, nlay); tsfc, coszen, adjes, scon
# and the albedo splits asdir/asdif/aldir/aldif (ncol,);
# icld/inflgsw/iceflgsw/liqflgsw/dyofyr are python ints SHARED by the
# batch (split batches by flag tuple).  DAY COLUMNS ONLY: coszen > 0 for
# every column, else the entry fails closed.  Arrays may be numpy or
# cupy; float32 is the contract dtype.  Non-f32 inputs are cast with
# numpy semantics on the host (exactly what the per-column driver's
# np.asarray(..., float32) does); float32 device copies/transposes are
# pure data movement and bit-preserving, so cupy f32 inputs never leave
# the device.
# ===========================================================================

#: Columns per chunk when no CUDA device can be asked (CPU-only imports
#: and host-side pricing with an unknown device): the width the chain
#: ran at before it was sized to the device (#310), kept so pricing
#: without a device stays deterministic.  It is NOT a ceiling: on a
#: device the width is the one that saturates that device's resident
#: threads (``batch_column_chunk``, woof.core.rrtmg_lw), narrowed only
#: when its priced transient would not fit the VRAM free at the time
#: (:func:`sw_batch_column_chunk`).  The former 2048 ceiling was
#: reasoned against 170 SMs x 1536 threads; a 170 SM part with 2048
#: threads per SM saturates at 3328 columns, and the ceiling cost it
#: 44 chunks where 27 cover the same columns, for no VRAM it needed to
#: protect: every part small enough for VRAM to matter saturates well
#: below 2048 columns.
SW_BATCH_COLUMN_CHUNK_NO_DEVICE = 2048


def sw_batch_free_device_bytes(cp=None, scratch=None):
    """Device bytes a batched call can allocate right now, or ``None``
    when no device can be asked: the driver's free bytes plus the pool's
    cached-free blocks plus whatever ``scratch`` already holds (its
    slots are reused, not re-allocated)."""
    try:
        if cp is None:
            import cupy as cp
        free = int(cp.cuda.runtime.memGetInfo()[0])
        pool = cp.get_default_memory_pool()
        free += int(pool.free_bytes())
        # A card can have room that this process's allocator cannot use.
        # Ignoring CUPY_GPU_MEMORY_LIMIT admitted a wide legacy radiation
        # chunk that failed before step 1 on a shared card. Cached blocks
        # count against the limit until reclaimed; live blocks cannot be.
        get_limit = getattr(pool, "get_limit", None)
        limit = int(get_limit()) if get_limit is not None else 0
        if limit:
            free = min(free, max(0, limit - int(pool.used_bytes())))
    except Exception:
        return None
    if scratch is not None:
        free += int(scratch.held_bytes())
    return free


def sw_vram_column_bound(nlay, free_bytes, upper, *, quantum=None):
    """Widest multiple of ``quantum`` at most ``upper`` (itself a
    quantum multiple) whose priced transient
    (:func:`sw_batched_vram_bytes`) fits ``free_bytes``; never below the
    quantum.  The pricing is monotone in the width, so the walk down
    from ``upper`` stops at the first fit."""
    from woof.core.rrtmg_lw import BATCH_CHUNK_QUANTUM
    quantum = int(BATCH_CHUNK_QUANTUM if quantum is None else quantum)
    nc = max(quantum, int(upper))
    while nc > quantum and sw_batched_vram_bytes(nc, nlay) > free_bytes:
        nc -= quantum
    return nc


def sw_batch_column_chunk(nlay=None, *, resident_threads=None,
                          free_bytes=None, quantum=None):
    """Columns per chunk of the batched SW chain.

    ``resident_threads`` is the device's resident-thread capacity (the
    current device's when None; 0 or an unusable value means no device
    and returns :data:`SW_BATCH_COLUMN_CHUNK_NO_DEVICE`).  The width is
    the smallest quantum multiple whose spcvmc launch (NGPTSW threads
    per column) covers the capacity; with ``nlay`` and ``free_bytes``
    both given it is narrowed to the widest quantum multiple whose
    priced transient fits (:func:`sw_vram_column_bound`).  Host-side
    pricing passes no ``free_bytes`` and so prices the saturation
    width, an upper bound on what a device run picks.  Width is
    workspace shape only: per-column results are bitwise identical at
    any width.
    """
    from woof.core.rrtmg_lw import (BATCH_CHUNK_QUANTUM,
                                     _device_resident_threads,
                                     batch_column_chunk)
    quantum = int(BATCH_CHUNK_QUANTUM if quantum is None else quantum)
    if resident_threads is None:
        resident_threads = _device_resident_threads()
    if not resident_threads or int(resident_threads) <= 0:
        return SW_BATCH_COLUMN_CHUNK_NO_DEVICE
    width = batch_column_chunk(NGPTSW, None, quantum=quantum,
                               resident_threads=resident_threads)
    if nlay is not None and free_bytes is not None:
        width = sw_vram_column_bound(nlay, int(free_bytes), width,
                                     quantum=quantum)
    return width


def __getattr__(name):
    if name == "SW_BATCH_COLUMN_CHUNK":
        return sw_batch_column_chunk()
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")

#: Workspace geometry of rsw_spcvmc_gpt[_b]: per thread,
#: SPCVMC_WK_ARRAYS float32 arrays and SPCVMC_WKC_ARRAYS uint8 arrays of
#: (nlayers + 1) entries.  KEEP IN STEP with RSW_SPCVMC_WK /
#: RSW_SPCVMC_WKC in kernels/rrtmg_sw.cu.
SPCVMC_WK_ARRAYS = 16
SPCVMC_WKC_ARRAYS = 0

#: Slot names of the batched chain's per-chunk workspace, by class.
SETCOEF_INT_SLOTS = ("jp", "jt", "jt1", "indself", "indfor", "tflag",
                     "lflag")
SETCOEF_REAL_SLOTS = ("colh2o", "colco2", "colo3", "coln2o", "colch4",
                      "colo2", "colmol", "co2mult", "selffac", "selffrac",
                      "forfac", "forfrac", "fac00", "fac01", "fac10",
                      "fac11")
SPCVMC_OUT_SLOTS = ("zcd", "zcu", "zfd", "zfu", "ztn", "ztcn")
SPC_ACCUM_SLOTS = ("pbbfd", "pbbfu", "pbbcd", "pbbcu", "pbbfddir",
                   "pbbcddir", "puvfd", "puvcd", "puvfddir", "puvcddir",
                   "pnifd", "pnicd", "pnifddir", "pnicddir")

#: The slots the kernels READ BEFORE WRITING, zeroed per chunk
#: (``SWBatchScratch.zeros``): wkl because rsw_inatm_layers_b scales all
#: RSW_NMOL species after writing only six of them (species 4 is read
#: unwritten). The call-zeroed err slot is written only on abort by
#: rsw_cldprmc_b and read once by the host after all chunks.
SW_CHUNK_ZEROED_SLOTS = ("wkl",)
SW_CALL_ZEROED_SLOTS = ("err",)

#: The slots that are constant kernel inputs, filled once when
#: allocated (``SWBatchScratch.constant``): the zero-aerosol optics
#: (ztaua, zasya = 0, zomga = 1) and the zero snow path of the non-5 ice
#: flag (cswpmc0, resnmc0), all ``const`` parameters of their kernels.
SW_CONSTANT_SLOTS = ("ztaua", "zasya", "zomga", "cswpmc0", "resnmc0")

#: Every ``SWBatchScratch.take`` slot: written in full by the kernel or
#: copy named before anything reads it, so it enters each chunk as the
#: previous chunk left it and is never zeroed.
SW_TAKE_SLOTS = (
    ("pdp", "rsw_inatm_layers_b, every (column, layer)"),
    ("coldry", "rsw_inatm_layers_b, every (column, layer)"),
    *((k, "rsw_setcoef_b, every (column, layer)")
      for k in SETCOEF_INT_SLOTS + SETCOEF_REAL_SLOTS),
    ("taug", "rsw_taumol_b, every (column, g-point, layer)"),
    ("taur", "rsw_taumol_b, every (column, g-point, layer)"),
    # True since rsw_sfluxzen_body stores on every path (WRF's zero where
    # a band's loop never reaches laysolfr).  Before, bands 16 and 27
    # returned without a store when laytrop == nlayers (model top below
    # ~191 hPa), so this slot handed spcvmc the previous chunk's or call's
    # bytes there: impossible surface fluxes on shallow or idealized
    # domains.  tests/test_rrtmg_sw_no_upper_flux_gpu.py fills the slot
    # with NaN, 1e30 or stale fluxes and holds the batched result to WRF.
    ("sflux", "rsw_sfluxzen_b, every (column, g-point)"),
    ("wk", "rsw_spcvmc_body: every entry read is written earlier by the "
           "same thread (fused layer pass and rsw_vrtqdr included)"),
    ("wkc", "zero-byte ABI placeholder; layer checks live in registers"),
    ("zincflx", "rsw_spcvmc_body, every (column, g-point)"),
    *((k, "rsw_spcvmc_body, every (column, g-point, level)")
      for k in SPCVMC_OUT_SLOTS),
    *((k, "rsw_spc_accum_b, stored once per (column, level) from a "
          "register sum") for k in SPC_ACCUM_SLOTS),
    ("swhr", "rsw_post_b, every (column, layer), both branches"),
    ("swhrc", "rsw_post_b, every (column, layer), both branches"),
)

#: Every workspace slot with its class: the audit surface for
#: tests/test_rrtmg_sw_workspace.py.
SW_SCRATCH_SLOTS = (
    tuple((k, "take", why) for k, why in SW_TAKE_SLOTS)
    + tuple((k, "constant", "const kernel input") for k in SW_CONSTANT_SLOTS)
    + tuple((k, "zeros", "read before write")
            for k in SW_CHUNK_ZEROED_SLOTS + SW_CALL_ZEROED_SLOTS)
)


class SWBatchScratch:
    """Per-chunk device workspace of the batched SW chain.

    ``take(slot, shape, dtype)`` returns a C-contiguous view of a byte
    buffer kept per slot, allocated (``xp.empty``) only when the slot is
    new or the request outgrows it, so across the chunks of a call and
    across calls a slot is allocated once and never zeroed: every
    ``take`` slot is fully written before it is read (SW_TAKE_SLOTS).
    ``constant(slot, shape, dtype, fill)`` is a kernel input filled once
    on allocation and never written; ``zeros(slot, shape, dtype)`` is a
    fresh ``xp.zeros`` per chunk for the read-before-write slots.  The
    counters are the test surface: ``allocations`` (buffers made),
    ``zeroed_bytes`` (memset per ``zeros``), ``filled_bytes`` (constant
    fills).  ``xp`` is cupy on the device and numpy in the CPU tests.
    """

    def __init__(self, xp):
        self.xp = xp
        self._bufs = {}
        self._consts = {}
        self.allocations = 0
        self.zeroed_bytes = 0
        self.filled_bytes = 0

    @staticmethod
    def _nbytes(shape, dtype):
        n = 1
        for s in shape:
            n *= int(s)
        return n * np.dtype(dtype).itemsize

    def take(self, slot, shape, dtype):
        shape = tuple(int(s) for s in shape)
        nbytes = self._nbytes(shape, dtype)
        buf = self._bufs.get(slot)
        if buf is None or buf.nbytes < nbytes:
            buf = self.xp.empty(nbytes, dtype=np.uint8)
            self._bufs[slot] = buf
            self.allocations += 1
        return buf[:nbytes].view(np.dtype(dtype)).reshape(shape)

    def constant(self, slot, shape, dtype, fill):
        shape = tuple(int(s) for s in shape)
        arr = self._consts.get(slot)
        if (arr is None or arr.dtype != np.dtype(dtype)
                or arr.shape[1:] != shape[1:] or arr.shape[0] < shape[0]):
            arr = self.xp.full(shape, fill, dtype=np.dtype(dtype))
            self._consts[slot] = arr
            self.allocations += 1
            self.filled_bytes += arr.nbytes
        return arr[:shape[0]]

    def zeros(self, slot, shape, dtype):
        shape = tuple(int(s) for s in shape)
        arr = self.xp.zeros(shape, dtype=np.dtype(dtype))
        self.zeroed_bytes += arr.nbytes
        return arr

    def held_bytes(self):
        return (sum(int(b.nbytes) for b in self._bufs.values())
                + sum(int(a.nbytes) for a in self._consts.values()))

    def slots(self):
        """Slot names currently held (take and constant)."""
        return tuple(self._bufs) + tuple(self._consts)

    def release(self):
        self._bufs.clear()
        self._consts.clear()

#: Every kernel in the SW translation unit (local-frame audit surface).
SW_GPU_KERNEL_NAMES = (
    "rsw_setcoef", "rsw_taumol", "rsw_sfluxzen", "rsw_cldprmc",
    "rsw_spcvmc_gpt", "rsw_spc_accum", "rsw_inatm_layers", "rsw_post",
    "rsw_setcoef_b", "rsw_taumol_b", "rsw_sfluxzen_b", "rsw_cldprmc_b",
    "rsw_spcvmc_gpt_b", "rsw_spc_accum_b", "rsw_inatm_layers_b",
    "rsw_post_b", "rsw_laysolfr_b",
)


def sw_batched_scratch_bytes(ncol_chunk, nlay):
    """Device bytes :class:`SWBatchScratch` holds once every slot has
    been taken at a chunk of ``ncol_chunk`` columns: the take slots and
    the constant slots (the zero snow pair is priced whether the ice
    flag uploads real snow or fills the constant, since exactly one of
    the two exists at a time and they are the same size).  Every term
    is rounded to CuPy's 512-byte pool quantum."""
    nc = int(ncol_chunk)
    nl = int(nlay)
    n1 = nl + 1
    f = 4
    s_nl = _r512(nc * nl * f)
    s_n1 = _r512(nc * n1 * f)
    s_g = _r512(nc * NGPTSW * f)
    s_gnl = _r512(nc * NGPTSW * nl * f)      # one (nc, 112, nl) f32 slab
    s_gn1 = _r512(nc * NGPTSW * n1 * f)
    s_bnl = _r512(nc * NBNDSW * nl * f)
    wk = _r512(nc * NGPTSW * SPCVMC_WK_ARRAYS * n1 * f)
    wkc = _r512(nc * NGPTSW * SPCVMC_WKC_ARRAYS * n1)
    return (2 * s_nl                         # pdp coldry
            + s_gnl + s_nl                   # cswpmc0 resnmc0 (or upload)
            + 23 * s_nl                      # 7 int + 16 real setcoef
            + 2 * s_gnl + s_g                # taug taur sflux
            + 3 * s_bnl                      # ztaua zasya zomga
            + wk + wkc
            + s_g                            # zincflx
            + 6 * s_gn1                      # zcd zcu zfd zfu ztn ztcn
            + 14 * s_n1                      # spc_accum outputs
            + 2 * s_nl)                      # swhr swhrc


def sw_batched_memset_bytes(ncol, nlay, column_chunk=None):
    """Device bytes the batched SW chain zeroes for ``ncol`` columns
    walked in chunks of ``column_chunk`` (default: all in one): the
    read-before-write slots only (SW_CHUNK_ZEROED_SLOTS: wkl per chunk
    and SW_CALL_ZEROED_SLOTS: the 4-byte abort flag per call) plus the two
    clean-sky output
    slabs that are zero by contract.  Unrounded, so it equals the
    ``SWBatchScratch.zeroed_bytes`` a call adds plus the two slabs."""
    ncol = int(ncol)
    nl = int(nlay)
    chunk = ncol if column_chunk is None else int(column_chunk)
    total = 0
    for c0 in range(0, ncol, chunk):
        nc = min(chunk, ncol - c0)
        total += nc * nl * MXMOL * 4
    return total + 4 + 2 * ncol * (nl + 1) * 4


def sw_batched_vram_bytes(ncol_chunk, nlay, ncol_total=None, *, mcica_layout="gpoint"):
    """Peak transient device bytes of ONE chunk of the batched SW chain.

    ``mcica_layout="column"`` prices resident contiguous float32 slabs;
    their caller-owned storage is outside this transient estimate.

    Derived from exactly the shapes rrtmg_sw_batched_device allocates:
    the workspace slots the scratch holds for the whole call
    (:func:`sw_batched_scratch_bytes`) plus the max over the three
    allocation high-water stages of the per-chunk uploads and transients
    that are not workspace (upload/inatm/cldprmc; setcoef/taumol/
    sfluxzen; spcvmc), plus the batch-level output slabs (priced at
    ncol_total, default = ncol_chunk).  Every term is rounded to CuPy's
    512-byte pool quantum, so the estimate tracks mempool.used_bytes()
    tightly (the accuracy test requires estimate >= measured >= 0.5 *
    estimate, measured from a released scratch).  The CudaSW instance
    constants (packed table buffer + ngb, a few MiB) are allocated at
    construction, before any batched call, and are NOT included.
    """
    if mcica_layout not in ("gpoint", "column"):
        raise ValueError("mcica_layout must be gpoint or column")
    copied = mcica_layout == "gpoint"
    nc = int(ncol_chunk)
    nl = int(nlay)
    n1 = nl + 1
    nt = nc if ncol_total is None else int(ncol_total)
    f = 4
    s_nl = _r512(nc * nl * f)
    s_n1 = _r512(nc * n1 * f)
    s_gnl = _r512(nc * NGPTSW * nl * f)      # one (nc, 112, nl) f32 slab
    s_b = _r512(nc * NBNDSW * f)
    s_m = _r512(nc * nl * MXMOL * f)         # wkl
    s_c = _r512(nc * f)
    # batch-level outputs, alive for the whole call
    out_b = 10 * _r512(nt * n1 * f) + 2 * _r512(nt * nl * f)
    # stage U: uploads + wkl + the 7 McICA slabs + taormc + radii + err
    stage_u = (s_n1                          # plev
               + 2 * s_nl                    # play tlay
               + 6 * s_nl                    # vmr x6
               + s_m                         # wkl (zeroed per chunk)
               + (7 * copied + 1) * s_gnl    # copied mcica + taormc
               + 2 * s_nl                    # reicmc relqmc
               + _r512(4))                   # err flag
    # stage S: the kept McICA slabs + laytrop + laysolfr (play/tlay/
    # coldry/wkl freed before the band kernels launch)
    stage_s = ((4 * copied + 1) * s_gnl       # copied cloud slabs + taormc
               + s_c + _r512(nc * NBNDSW * f))  # laytrop + laysolfr
    # stage P: cloud slabs stay alive through spcvmc along with scalar uploads.
    stage_p = (4 * copied + 1) * s_gnl + 3 * s_b + s_c
    return (sw_batched_scratch_bytes(nc, nl)
            + max(stage_u, stage_s, stage_p) + out_b)


def sw_batched_to_host(out):
    """Host (numpy) copies of an rrtmg_sw_batched_device result, plus the
    three derived difference outputs the per-column driver computes on
    the host (swdkdif, sibvisdif, sibnirdif).  numpy float32 elementwise
    subtraction is the identical single IEEE operation the per-column
    F(a - b) scalar loop performs, including subnormal handling (x86
    keeps full IEEE semantics host-side)."""
    import cupy as cp
    h = {k: cp.asnumpy(v) for k, v in out.items()}
    h["swdkdif"] = h["swdflx"] - h["swdkdir"]
    h["sibvisdif"] = h["puvfd"] - h["sibvisdir"]
    h["sibnirdif"] = h["pnifd"] - h["sibnirdir"]
    del h["puvfd"], h["pnifd"]
    return h


def validate_aer1_optics_tables(tables):
    """Check the existing rrsw_aer tables against the climatology manifest."""
    import hashlib
    for name in ("rsrtaua", "rsrpiza", "rsrasya"):
        value = np.ascontiguousarray(getattr(tables, name), dtype=F)
        if value.shape != (14, 6) or hashlib.sha256(value.tobytes()).hexdigest() != tables.aer1_optics_sha256[name]:
            raise ValueError(f"aer_opt=1 rrsw_aer {name} manifest hash mismatch; altered optics would change shortwave flux and heating")
