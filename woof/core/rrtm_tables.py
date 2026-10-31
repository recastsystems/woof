# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
#   RRTM longwave, transcribed from WRF v4.6.1 phys/module_ra_rrtm.F.
#   That file carries AER's authorship -- Eli J. Mlawer, Steven J.
#   Taubman, Shepard A. Clough, Atmospheric & Environmental Research, Inc.
#   -- but no copyright line and no licence clause, so it falls under
#   WRF's public-domain declaration.  AER is credited here as a matter of
#   accuracy rather than of condition.
# ======================================================================
"""g-point reduction and lookup tables for WRF v4.6.1 RRTM longwave.

This module is the Python transcription of ``rrtm_lookuptable``'s
post-READ half plus ``CMBGB1``--``CMBGB16``
(``phys/module_ra_rrtm.F:7486-7630`` and ``:2302-3523``).  It takes the
two frozen data sources that :mod:`woof.core.rrtm` and
``tools/build_rrtm_lw_statics.py`` already package --

* ``woof/data/wrf_radiation/RRTM_DATA`` -- the 16 big-endian sequential
  records of 256-g-point absorption coefficients, and
* ``woof/data/wrf_radiation/rrtm_lw_statics.npz`` -- every module-head
  ``DATA`` statement (g-point bookkeeping, Planck integrals, reference
  profiles, band edges)

-- and produces the 140-g-point *combined* tables the ``TAUGBn`` routines
read (``ABSAn``/``ABSBn``/``SELFREFCn``/``FRACREFACn``/...), together
with the three Pade lookup tables (``TAU``/``TF``/``TRANS``) and the two
band-2 correction tables (``CORR1``/``CORR2``).

Arithmetic fidelity: WRF accumulates every ``SUMK``/``SUMF`` in default
``REAL`` (FP32) by sequential addition over at most eight members.  The
reduction here does the same -- an explicit FP32 running sum, never a
pairwise ``np.sum`` -- so the combined coefficients match the Fortran
reduction order.  Array flattening follows the Fortran subscript
expressions exactly (see ``_ABSA_SHAPES``/``_ABSB_SHAPES``), which is
Fortran element order over the leading dimensions.

The build is deterministic and pure; :func:`load_rrtm_lw_tables` caches
the result for the process.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from woof.core.rrtm import (DATA_DIR, RRTM_NBANDS, RRTM_NGPT,
                             load_rrtm_raw_tables)


STATICS_PATH = DATA_DIR / "rrtm_lw_statics.npz"
#: sha256 of the packaged statics archive, pinned by
#: tools/build_rrtm_lw_statics.py's receipt.
RRTM_LW_STATICS_SHA256 = \
    "1fa277fe31b5412434c7656f18913b14daccaf32e64aa2617" \
    "6a4b9a808ccf953"

#: module_ra_rrtm.F:8-23.  Combined g-point counts per band (== NGC).
RRTM_MG = 16
#: module_ra_rrtm.F:29.  ``ONEMINUS = 1. - 1.E-6``.
RRTM_ONEMINUS = np.float32(1.0 - 1.0e-6)
#: module_ra_rrtm.F:6195-6265 (RTRN) and :6645 (GASABS).
RRTM_SECANG = np.float32(1.66)
RRTM_WTNUM = np.float32(0.5)
#: module_ra_rrtm.F:7499.  ``BPADE = 1./0.278``: two REAL literals, so the
#: quotient is formed in float32 (3.5971224), not rounded from a binary64
#: quotient (3.5971222, one ULP low).  The WRF column oracle
#: (tools/rrtm_lw_wrf461_oracle) measured the binary64 spelling one ULP off
#: BPADE itself and up to 3 ULP off 3,575 of the 5,001 TAU entries.
RRTM_BPADE = np.float32(1.0) / np.float32(0.278)
#: module_ra_rrtm.F:6771-6773.  ``PI = 2.*ASIN(1.)`` is REAL (gfortran folds
#: ASIN(1.) to the float32 pi/2), then ``FLUXFAC = PI * 2.D4`` widens that
#: float32 pi to binary64, multiplies and stores into the REAL FLUXFAC.  The
#: binary64 pi the old spelling used gives 62831.85, one ULP below WRF's
#: 62831.855, and that ULP reached every flux.
_PI_F32 = np.float32(2.0) * np.float32(np.pi / 2.0)
RRTM_FLUXFAC = np.float32(float(_PI_F32) * 2.0e4)

# Fortran subscript expressions of the CMBGBn ABSA/ABSB writes, expressed
# as the leading-dimension shape that is flattened in Fortran order.
#   band 1  CMBGB1 :2337  ABSA1(JTJT+(JPJP-1)*5, IGC)          -> (5,13)
#   band 3  CMBGB3 :2497  ABSA3(JN+(JTJT-1)*10+(JPJP-1)*50,..) -> (10,5,13)
#   band 4  CMBGB4 :2589  ABSA4(JN+(JTJT-1)*9+(JPJP-1)*45,..)  -> (9,5,13)
#   band 8  CMBGB8 :2926  ABSA8(JTJT+(JPJP-1)*5, IGC)          -> (5,7)
#   band 9  CMBGB9 :3017  ABSA9(JN+(JTJT-1)*11+(JPJP-1)*55,..) -> (11,5,13)
_ABSA_SHAPES = {
    1: (5, 13), 2: (5, 13), 3: (10, 5, 13), 4: (9, 5, 13), 5: (9, 5, 13),
    6: (5, 13), 7: (9, 5, 13), 8: (5, 7), 9: (11, 5, 13), 10: (5, 13),
    11: (5, 13), 12: (9, 5, 13), 13: (9, 5, 13), 14: (5, 13),
    15: (9, 5, 13), 16: (9, 5, 13),
}
#   band 1  CMBGB1 :2349  ABSB1(JTJT+(JPJP-13)*5, IGC)         -> (5,47)
#   band 3  CMBGB3 :2511  ABSB3(JN+(JTJT-1)*5+(JPJP-13)*25,..) -> (5,5,47)
#   band 4  CMBGB4 :2603  ABSB4(JN+(JTJT-1)*6+(JPJP-13)*30,..) -> (6,5,47)
#   band 8  CMBGB8 :2939  ABSB8(JTJT+(JPJP-7)*5, IGC)          -> (5,53)
_ABSB_SHAPES = {
    1: (5, 47), 2: (5, 47), 3: (5, 5, 47), 4: (6, 5, 47), 5: (5, 5, 47),
    7: (5, 47), 8: (5, 53), 9: (5, 47), 10: (5, 47), 11: (5, 47),
    14: (5, 47),
}
#: Bands whose CMBGBn reduces a SELFREF(10,MG) block.  Band 10 has none
#: (CMBGB10 takes no SELFREF argument, module_ra_rrtm.F:3087-3092).
_SELFREF_BANDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16)
#: Weighted 16-vector extras: statics name -> (band, combined name).
_SCALAR_WEIGHTED = {
    "FORREF1": (1, "FORREFC1"), "FORREF2": (2, "FORREFC2"),
    "FORREF3": (3, "FORREFC3"),
    "ABSN2OA3": (3, "ABSN2OAC3"), "ABSN2OB3": (3, "ABSN2OBC3"),
    "CCL45": (5, "CCL4C5"),
    "ABSCO26": (6, "ABSCO2C6"), "CFC11ADJ6": (6, "CFC11ADJC6"),
    "CFC126": (6, "CFC12C6"),
    "ABSCO27": (7, "ABSCO2C7"),
    "ABSCO2A8": (8, "ABSCO2AC8"), "ABSCO2B8": (8, "ABSCO2BC8"),
    "ABSN2OA8": (8, "ABSN2OAC8"), "ABSN2OB8": (8, "ABSN2OBC8"),
    "CFC128": (8, "CFC12C8"), "CFC22ADJ8": (8, "CFC22ADJC8"),
}
#: Unweighted Planck-fraction reductions: statics name ->
#: (band, combined name, trailing shape).  A trailing shape of () is a
#: bare MG vector; (n,) is the Fortran ``FRACREF*(MG,n)`` block.
_FRACREF = {
    "FRACREFA1": (1, "FRACREFAC1", ()),
    "FRACREFB1": (1, "FRACREFBC1", ()),
    "FRACREFA2": (2, "FRACREFAC2", (13,)),
    "FRACREFB2": (2, "FRACREFBC2", ()),
    "FRACREFA3": (3, "FRACREFAC3", (10,)),
    "FRACREFB3": (3, "FRACREFBC3", (5,)),
    "FRACREFA4": (4, "FRACREFAC4", (9,)),
    "FRACREFB4": (4, "FRACREFBC4", (6,)),
    "FRACREFA5": (5, "FRACREFAC5", (9,)),
    "FRACREFB5": (5, "FRACREFBC5", (5,)),
    "FRACREFA6": (6, "FRACREFAC6", ()),
    "FRACREFA7": (7, "FRACREFAC7", (9,)),
    "FRACREFB7": (7, "FRACREFBC7", ()),
    "FRACREFA8": (8, "FRACREFAC8", ()),
    "FRACREFB8": (8, "FRACREFBC8", ()),
    "FRACREFA9": (9, "FRACREFAC9", (9,)),
    "FRACREFB9": (9, "FRACREFBC9", ()),
    "FRACREFA10": (10, "FRACREFAC10", ()),
    "FRACREFB10": (10, "FRACREFBC10", ()),
    "FRACREFA11": (11, "FRACREFAC11", ()),
    "FRACREFB11": (11, "FRACREFBC11", ()),
    "FRACREFA12": (12, "FRACREFAC12", (9,)),
    "FRACREFA13": (13, "FRACREFAC13", (9,)),
    "FRACREFA14": (14, "FRACREFAC14", ()),
    "FRACREFB14": (14, "FRACREFBC14", ()),
    "FRACREFA15": (15, "FRACREFAC15", (9,)),
    "FRACREFA16": (16, "FRACREFAC16", (9,)),
}


@dataclass(frozen=True)
class RRTMLongwaveTables:
    """Combined 140-g-point coefficient set plus RRTM's lookup tables.

    ``absa``/``absb`` are keyed by 1-based band number and shaped
    ``(nrows, ngc[band])``; ``nrows`` is the Fortran flattened extent of
    the reference-atmosphere/temperature/pressure block.  Everything is
    read-only native FP32.
    """

    absa: dict[int, np.ndarray]
    absb: dict[int, np.ndarray]
    selfrefc: dict[int, np.ndarray]
    combined: dict[str, np.ndarray]
    statics: dict[str, np.ndarray]
    tau: np.ndarray
    tf: np.ndarray
    trans: np.ndarray
    corr1: np.ndarray
    corr2: np.ndarray
    rwgt: np.ndarray

    # --- bookkeeping shortcuts (module_ra_rrtm.F "data 2"/"data 6") ---
    @property
    def ngc(self) -> np.ndarray:
        return self.statics["NGC"]

    @property
    def ngs(self) -> np.ndarray:
        return self.statics["NGS"]

    @property
    def ngb(self) -> np.ndarray:
        return self.statics["NGB"]

    @property
    def nspa(self) -> np.ndarray:
        return self.statics["NSPA"]

    @property
    def nspb(self) -> np.ndarray:
        return self.statics["NSPB"]


def _group_spans(band: int, ngc: np.ndarray, ngs: np.ndarray,
                 ngn: np.ndarray) -> tuple[tuple[int, int], ...]:
    """``NGN(NGS(band-1)+IGC)`` partition of a band's 16 input g-points.

    Transcribes the ``DO IPR = 1, NGN(NGS(n-1)+IGC)`` bounds shared by
    every CMBGBn (module_ra_rrtm.F:2331, 2404, 2487, ...).
    """
    base = 0 if band == 1 else int(ngs[band - 2])
    spans = []
    start = 0
    for igc in range(int(ngc[band - 1])):
        count = int(ngn[base + igc])
        spans.append((start, start + count))
        start += count
    if start != RRTM_MG:
        raise ValueError(
            f"band {band}: NGN partition covers {start} of {RRTM_MG} "
            "g-points")
    return tuple(spans)


def _reduce_g(source: np.ndarray, spans, weights) -> np.ndarray:
    """FP32 sequential ``SUMK``/``SUMF`` reduction over the last axis.

    ``source`` has trailing axis 16 (the input g-points).  ``weights`` is
    either the band's 16 ``RWGT`` entries (the weighted ``SUMK`` form) or
    ``None`` for the unweighted Planck-fraction ``SUMF`` form.  The
    accumulation is a running FP32 add in the Fortran's IPR order.
    """
    lead = source.shape[:-1]
    out = np.zeros(lead + (len(spans),), dtype=np.float32)
    for igc, (start, stop) in enumerate(spans):
        total = np.zeros(lead, dtype=np.float32)
        for iprsm in range(start, stop):
            term = source[..., iprsm]
            if weights is not None:
                term = (term * weights[iprsm]).astype(np.float32)
            total = (total + term).astype(np.float32)
        out[..., igc] = total
    return out


def _fortran_flatten(array: np.ndarray, lead_shape) -> np.ndarray:
    """Collapse ``lead_shape`` in Fortran element order, keep the g axis."""
    rows = int(np.prod(lead_shape))
    if array.shape[:-1] != tuple(lead_shape):
        raise ValueError(
            f"expected leading shape {tuple(lead_shape)}, got "
            f"{array.shape[:-1]}")
    return np.ascontiguousarray(
        array.reshape((rows, array.shape[-1]), order="F"))


def _build_rwgt(statics: dict[str, np.ndarray]) -> np.ndarray:
    """Transcribe the ``DO 500 IBND`` weighting loop (F:7524-7550).

    ``RWGT(IND) = WT(IG)/WTSM(NGM(IND))`` for combined bands, 1.0 for
    bands whose ``NGC`` is already 16.  Returned 0-based over 256 entries.
    """
    ngc = statics["NGC"]
    ngn = statics["NGN"]
    ngm = statics["NGM"]
    ng = statics["NG"]
    wt = statics["WT"]
    rwgt = np.zeros(RRTM_MG * RRTM_NBANDS, dtype=np.float32)
    igcsm = 0
    for ibnd in range(1, RRTM_NBANDS + 1):
        iprsm = 0
        if int(ngc[ibnd - 1]) < RRTM_MG:
            wtsm = np.zeros(RRTM_MG, dtype=np.float32)
            for igc in range(int(ngc[ibnd - 1])):
                wtsum = np.float32(0.0)
                for _ in range(int(ngn[igcsm])):
                    wtsum = np.float32(wtsum + wt[iprsm])
                    iprsm += 1
                wtsm[igc] = wtsum
                igcsm += 1
            for ig in range(int(ng[ibnd - 1])):
                ind = (ibnd - 1) * RRTM_MG + ig
                rwgt[ind] = np.float32(wt[ig] / wtsm[int(ngm[ind]) - 1])
        else:
            for ig in range(int(ng[ibnd - 1])):
                rwgt[(ibnd - 1) * RRTM_MG + ig] = np.float32(1.0)
                igcsm += 1
    if igcsm != RRTM_NGPT:
        raise ValueError(
            f"g-point reduction visited {igcsm} groups, expected "
            f"{RRTM_NGPT}")
    return rwgt


def _pade_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``TAU``/``TRANS``/``TF`` lookup tables (F:7493-7509).

    Endpoints are set explicitly by WRF; the 1..4999 interior is the
    Pade form with the ``TAU < 0.1`` linear-in-tau replacement.
    """
    tau = np.zeros(5001, dtype=np.float32)
    trans = np.zeros(5001, dtype=np.float32)
    tf = np.zeros(5001, dtype=np.float32)
    tau[0] = np.float32(0.0)
    tau[5000] = np.float32(1.0e10)
    trans[0] = np.float32(1.0)
    trans[5000] = np.float32(0.0)
    tf[0] = np.float32(0.0)
    tf[5000] = np.float32(1.0)
    itre = np.arange(1, 5000, dtype=np.float32)
    tfn = (itre / np.float32(5.0e3)).astype(np.float32)
    interior = (RRTM_BPADE * tfn / (np.float32(1.0) - tfn)).astype(np.float32)
    tau[1:5000] = interior
    # TRANS(ITRE) = EXP(-TAU(ITRE)) is gfortran's float32 EXP, the libm
    # expf WOOF carries as its own (woof.core.noahmp_libm.expf).  NumPy's
    # float32 exp is a different function (its own vector loop on AVX-512
    # hosts); the oracle measured it up to 127 ULP off TRANS.
    from woof.core.noahmp_libm import expf
    trans_interior = np.fromiter((expf(-value) for value in interior),
                                 dtype=np.float32, count=interior.size)
    trans[1:5000] = trans_interior
    linear = (interior / np.float32(6.0)).astype(np.float32)
    full = (np.float32(1.0) - np.float32(2.0)
            * ((np.float32(1.0) / interior)
               - (trans_interior / (np.float32(1.0) - trans_interior)))
            ).astype(np.float32)
    tf[1:5000] = np.where(interior < np.float32(0.1), linear, full)
    return tau, tf, trans


def _corr_tables() -> tuple[np.ndarray, np.ndarray]:
    """Band-2 ``CORR1``/``CORR2`` lookup tables (F:7510-7518)."""
    corr1 = np.zeros(201, dtype=np.float32)
    corr2 = np.zeros(201, dtype=np.float32)
    corr1[0] = np.float32(1.0)
    corr1[200] = np.float32(1.0)
    corr2[0] = np.float32(1.0)
    corr2[200] = np.float32(1.0)
    index = np.arange(1, 200, dtype=np.float32)
    fp = (np.float32(0.005) * index).astype(np.float32)
    rtfp = np.sqrt(fp).astype(np.float32)
    corr1[1:200] = (rtfp / fp).astype(np.float32)
    corr2[1:200] = ((np.float32(1.0) - rtfp)
                    / (np.float32(1.0) - fp)).astype(np.float32)
    return corr1, corr2


def load_rrtm_lw_statics() -> dict[str, np.ndarray]:
    """Read the packaged module-head ``DATA`` archive as read-only arrays."""
    from woof.tables import table_path
    with np.load(table_path("rrtm", STATICS_PATH)) as archive:
        statics = {name: np.array(archive[name]) for name in archive.files}
    for array in statics.values():
        array.setflags(write=False)
    return statics


@lru_cache(maxsize=1)
def load_rrtm_lw_tables() -> RRTMLongwaveTables:
    """Build the 140-g-point combined tables (``rrtm_lookuptable`` tail)."""
    statics = load_rrtm_lw_statics()
    raw = load_rrtm_raw_tables().arrays
    rwgt = _build_rwgt(statics)
    ngc, ngs, ngn = statics["NGC"], statics["NGS"], statics["NGN"]
    spans = {band: _group_spans(band, ngc, ngs, ngn)
             for band in range(1, RRTM_NBANDS + 1)}

    def band_weights(band: int) -> np.ndarray:
        start = (band - 1) * RRTM_MG
        return rwgt[start:start + RRTM_MG]

    absa: dict[int, np.ndarray] = {}
    absb: dict[int, np.ndarray] = {}
    selfrefc: dict[int, np.ndarray] = {}
    combined: dict[str, np.ndarray] = {}

    for band, shape in _ABSA_SHAPES.items():
        source = _fortran_flatten(raw[f"abscoefL{band}"], shape)
        absa[band] = _reduce_g(source, spans[band], band_weights(band))
    for band, shape in _ABSB_SHAPES.items():
        source = _fortran_flatten(raw[f"abscoefH{band}"], shape)
        absb[band] = _reduce_g(source, spans[band], band_weights(band))
    for band in _SELFREF_BANDS:
        source = raw[f"selfref{band}"]  # (10, MG)
        selfrefc[band] = _reduce_g(source, spans[band], band_weights(band))

    for name, (band, target) in _SCALAR_WEIGHTED.items():
        combined[target] = _reduce_g(
            statics[name], spans[band], band_weights(band))
    for name, (band, target, trailing) in _FRACREF.items():
        source = statics[name]
        if trailing:
            # Fortran FRACREF*(MG, n): MG fastest.
            source = source.reshape((RRTM_MG,) + trailing, order="F")
            source = np.ascontiguousarray(np.moveaxis(source, 0, -1))
        combined[target] = _reduce_g(source, spans[band], None)

    # CMBGB9's ABSN2O block, F:3044-3060: ABSN2O(JND+IPRSM) with
    # JND=(JN-1)*16 reduces into ABSN2OC(JNDC+IGC), JNDC=(JN-1)*NGC(9).
    # Both are (JN-major, g-minor), i.e. C order with JN first.
    absn2o9 = statics["ABSN2O9"].reshape((3, RRTM_MG))
    combined["ABSN2OC9"] = _reduce_g(absn2o9, spans[9], band_weights(9))

    tau, tf, trans = _pade_tables()
    corr1, corr2 = _corr_tables()

    for array in (*absa.values(), *absb.values(), *selfrefc.values(),
                  *combined.values(), tau, tf, trans, corr1, corr2, rwgt):
        array.setflags(write=False)
    return RRTMLongwaveTables(
        absa=absa, absb=absb, selfrefc=selfrefc, combined=combined,
        statics=statics, tau=tau, tf=tf, trans=trans, corr1=corr1,
        corr2=corr2, rwgt=rwgt)


__all__ = ["RRTMLongwaveTables", "RRTM_BPADE", "RRTM_FLUXFAC", "RRTM_MG",
           "RRTM_ONEMINUS", "RRTM_SECANG", "RRTM_WTNUM",
           "RRTM_LW_STATICS_SHA256", "STATICS_PATH",
           "load_rrtm_lw_statics", "load_rrtm_lw_tables"]
