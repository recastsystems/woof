"""Noah-MP snowpack layer routines, transcribed from WRF v4.6.1 in FP32.

Ports ``MODULE_SF_NOAHMPLSM``'s SNOWFALL, COMPACT, COMBINE, DIVIDE, COMBO and
SNOWH2O, plus the SNOWWATER driver that sequences them, from
``phys/module_sf_noahmplsm.F`` at WRF commit
``d66e442fccc04111067e29274c9f9eaccc3cef28``
(``sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282``),
lines 6398-7230.

``kind_phys == kind(1.0)`` in that build, so every quantity here is IEEE
binary32 and every arithmetic boundary is pinned to ``numpy.float32``.  All
literals are float32 constants rather than Python floats, so the port does not
depend on which NumPy scalar-promotion rules are in force.

Fidelity notes that are easy to get wrong and are therefore pinned by the
fixture rather than by comment alone:

* Snow layer indices run Fortran-style, ``-NSNOW+1`` (top) to ``0`` (bottom),
  with soil layers ``1..NSOIL``.  :class:`SnowColumn` keeps that convention.
* ``DZSNSO`` is a *positive* thickness on entry to and exit from every leaf
  here; SNOWWATER's double negation is an internal detail of rebuilding
  ``ZSNSO``.
* COMBINE declares ``PONDING1``/``PONDING2`` ``INTENT(OUT)`` but assigns them
  on only some paths, and gfortran passes scalar dummies by reference, so an
  unassigned path leaves the caller's value standing.  :func:`combine` and
  :func:`snowh2o` therefore take both as arguments and return them, rather
  than inventing a zero.
* DIVIDE's ``DZ``/``SWICE``/``SWLIQ``/``TSNO`` slots above ``ABS(ISNOW)`` are
  conditionally assigned locals.  They are poisoned with NaN here; the oracle
  build proves no emitted value depends on them by re-running the whole
  fixture under ``-finit-real=snan`` and requiring byte-identical output.

COMPACT is the only leaf in this file that evaluates a transcendental.  It
calls :func:`woof.core.noahmp_libm.expf`, the verified glibc transcription,
because neither ``numpy.exp`` on float32 nor "compute in float64 and round
once" reproduces glibc's ``expf`` bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from woof.core.noahmp_libm import expf

__all__ = [
    "NSNOW_DEFAULT",
    "NSOIL_DEFAULT",
    "SnowColumn",
    "combo",
    "snowfall",
    "compact",
    "combine",
    "divide",
    "snowh2o",
    "snowwater",
    "COMPACTION_OPTIONS",
    "COMPACTION_LABEL",
    "compaction_cbd",
]


def _f(x) -> np.float32:
    return np.float32(x)


NSNOW_DEFAULT = 3
NSOIL_DEFAULT = 4

# --- module_sf_noahmplsm.F lines 207-220, the shared physical constants -----
TFRZ = _f(273.16)      # freezing/melting point (K)
HFUS = _f(0.3336e06)   # latent heat of fusion (J/kg)
CWAT = _f(4.188e06)    # volumetric heat capacity of water (J/m3/K)
CICE = _f(2.094e06)    # volumetric heat capacity of ice (J/m3/K)
DENH2O = _f(1000.0)    # density of water (kg/m3)
DENICE = _f(917.0)     # density of ice (kg/m3)

# --- COMPACT's own PARAMETERs, lines 7000-7008 -----------------------------
_C2 = _f(21.0e-3)      # [m3/kg]
_C3 = _f(2.5e-6)       # [1/s]
_C4 = _f(0.04)         # [1/K]
_C5 = _f(2.0)
_DM = _f(100.0)        # upper limit on destructive metamorphism [kg/m3]
_ETA0 = _f(1.33e+6)    # viscosity coefficient, He et al. 2021 value

# --- Snow compaction option 2: overburden C_bd(T_air, p_sfc) --------------
# Re-implemented from published descriptions, not bitwise to upstream.
# Source: Abolafia-Rosenzweig, He, Chen & Barlage (2024), "Evaluating and
# Enhancing Snow Compaction Process in the Noah-MP Land Surface Model",
# JAMES 16, e2023MS003869, doi:10.1029/2023MS003869 (CC BY), Section 2.4:
#   Eq. 11  C_bd = -0.00069503 * T_air + 0.20606699      (T_air in K)
#   Eq. 12  pressure-binned C_bd,min: 0.019 (psfc < 80,000 Pa),
#           0.018 (80,000 <= psfc < 85,000 Pa), 0.017 (psfc >= 85,000 Pa)
#   text    upper constraint C_bd <= 0.0315 (T_air below ~251 K)
# Eq. 12 is printed with "min(C_bd, .)", but the same section calls the
# bins "variable lower constraints" and "C_bd,min", and Figure 2's three
# lines are Eq. 11 floored at 0.019/0.018/0.017 at warm temperatures, so
# the bins are applied as floors (max).  No upstream source file was read.
COMPACTION_OPTIONS = (1, 2)
COMPACTION_LABEL = ("re-implemented from published descriptions, "
                    "not bitwise to upstream")
_AR24_SLOPE = _f(-0.00069503)   # [m3/kg/K]  Eq. 11
_AR24_ICEPT = _f(0.20606699)    # [m3/kg]    Eq. 11
_AR24_P_LO = _f(80000.0)        # [Pa]       Eq. 12 bin edge
_AR24_P_HI = _f(85000.0)        # [Pa]       Eq. 12 bin edge
_AR24_MIN_LO = _f(0.019)        # psfc < 80 kPa (high terrain)
_AR24_MIN_MID = _f(0.018)       # 80 <= psfc < 85 kPa
_AR24_MIN_HI = _f(0.017)        # psfc >= 85 kPa
_AR24_MAX = _f(0.0315)          # upper constraint, Section 2.4 text


def compaction_cbd(sfctmp, psfc) -> np.float32:
    """Overburden compaction parameter C_bd [m3/kg] of compaction option 2.

    ``sfctmp`` is the surface air temperature Noah-MP is forced with (K) and
    ``psfc`` the surface pressure (Pa).  Float32, one rounding per operation,
    in the order the CUDA option kernel evaluates it.
    """
    t = _f(sfctmp)
    p = _f(psfc)
    cbd = _f(_f(_AR24_SLOPE * t) + _AR24_ICEPT)
    if p < _AR24_P_LO:
        floor = _AR24_MIN_LO
    elif p < _AR24_P_HI:
        floor = _AR24_MIN_MID
    else:
        floor = _AR24_MIN_HI
    return _f(min(max(cbd, floor), _AR24_MAX))


# --- COMBINE's DZMIN DATA statement, line 6648 ------------------------------
# MB's revised limits.  1-based like the Fortran MSSI index.
_DZMIN = (_f(0.025), _f(0.025), _f(0.1))

# --- SNOWH2O's max_liq_mass_fraction, line 7135 ----------------------------
_MAX_LIQ_MASS_FRACTION = _f(0.4)

_ZERO = _f(0.0)
_ONE = _f(1.0)
_HALF = _f(0.5)


class _Col:
    """A float32 vector addressed with Fortran's own lower bound.

    Keeps the transcription's subscripts identical to the Fortran so a reader
    can diff the two line by line instead of re-deriving offsets.
    """

    __slots__ = ("data", "lo")

    def __init__(self, data: np.ndarray, lo: int):
        self.data = data
        self.lo = lo

    def __getitem__(self, j: int) -> np.float32:
        return self.data[j - self.lo]

    def __setitem__(self, j: int, v) -> None:
        self.data[j - self.lo] = v


@dataclass
class SnowColumn:
    """Snow/soil column state in WRF's own index convention and precision."""

    nsnow: int = NSNOW_DEFAULT
    nsoil: int = NSOIL_DEFAULT
    isnow: int = 0
    snowh: np.float32 = _ZERO
    sneqv: np.float32 = _ZERO
    snice: np.ndarray = field(default=None)   # (-nsnow+1 : 0)
    snliq: np.ndarray = field(default=None)   # (-nsnow+1 : 0)
    stc: np.ndarray = field(default=None)     # (-nsnow+1 : nsoil)
    zsnso: np.ndarray = field(default=None)   # (-nsnow+1 : nsoil)
    dzsnso: np.ndarray = field(default=None)  # (-nsnow+1 : nsoil)
    sh2o: np.ndarray = field(default=None)    # (1 : nsoil)
    sice: np.ndarray = field(default=None)    # (1 : nsoil)

    def __post_init__(self):
        n_s, n_f = self.nsnow, self.nsnow + self.nsoil
        for name, n in (("snice", n_s), ("snliq", n_s), ("stc", n_f),
                        ("zsnso", n_f), ("dzsnso", n_f),
                        ("sh2o", self.nsoil), ("sice", self.nsoil)):
            v = getattr(self, name)
            setattr(self, name, np.zeros(n, dtype=np.float32) if v is None
                    else np.asarray(v, dtype=np.float32).copy())
        self.snowh = _f(self.snowh)
        self.sneqv = _f(self.sneqv)
        self.isnow = int(self.isnow)

    # Fortran-indexed views.  Cheap objects; rebuilt per access so that a
    # caller replacing an array attribute is never left with a stale view.
    @property
    def SNICE(self) -> _Col:
        return _Col(self.snice, -self.nsnow + 1)

    @property
    def SNLIQ(self) -> _Col:
        return _Col(self.snliq, -self.nsnow + 1)

    @property
    def STC(self) -> _Col:
        return _Col(self.stc, -self.nsnow + 1)

    @property
    def ZSNSO(self) -> _Col:
        return _Col(self.zsnso, -self.nsnow + 1)

    @property
    def DZSNSO(self) -> _Col:
        return _Col(self.dzsnso, -self.nsnow + 1)

    @property
    def SH2O(self) -> _Col:
        return _Col(self.sh2o, 1)

    @property
    def SICE(self) -> _Col:
        return _Col(self.sice, 1)

    def copy(self) -> "SnowColumn":
        return SnowColumn(
            nsnow=self.nsnow, nsoil=self.nsoil, isnow=self.isnow,
            snowh=self.snowh, sneqv=self.sneqv,
            snice=self.snice, snliq=self.snliq, stc=self.stc,
            zsnso=self.zsnso, dzsnso=self.dzsnso,
            sh2o=self.sh2o, sice=self.sice,
        )


# ===========================================================================
# COMBO -- module_sf_noahmplsm.F lines 6920-6970
# ===========================================================================

def combo(dz, wliq, wice, t, dz2, wliq2, wice2, t2):
    """Combine two snow elements into one; returns the updated element 1.

    Element 2 is INTENT(IN) in WRF and is not returned.
    """
    dz, wliq, wice, t = _f(dz), _f(wliq), _f(wice), _f(t)
    dz2, wliq2, wice2, t2 = _f(dz2), _f(wliq2), _f(wice2), _f(t2)

    dzc = dz + dz2
    wicec = wice + wice2
    wliqc = wliq + wliq2
    h = (CICE * wice + CWAT * wliq) * (t - TFRZ) + HFUS * wliq
    h2 = (CICE * wice2 + CWAT * wliq2) * (t2 - TFRZ) + HFUS * wliq2

    hc = h + h2
    if hc < _ZERO:
        tc = TFRZ + hc / (CICE * wicec + CWAT * wliqc)
    elif hc <= HFUS * wliqc:
        tc = TFRZ
    else:
        tc = TFRZ + (hc - HFUS * wliqc) / (CICE * wicec + CWAT * wliqc)

    return dzc, wliqc, wicec, tc


# ===========================================================================
# SNOWFALL -- lines 6539-6606
# ===========================================================================

def snowfall(col: SnowColumn, dt, qsnow, snowhin, sfctmp) -> None:
    """Account for new snowfall; may create the first snow layer (0 -> -1)."""
    dt, qsnow = _f(dt), _f(qsnow)
    snowhin, sfctmp = _f(snowhin), _f(sfctmp)
    DZSNSO, STC, SNICE, SNLIQ = col.DZSNSO, col.STC, col.SNICE, col.SNLIQ

    newnode = 0

    # shallow snow / no layer
    if col.isnow == 0 and qsnow > _ZERO:
        col.snowh = col.snowh + snowhin * dt
        col.sneqv = col.sneqv + qsnow * dt

    # creating a new layer.  C.He removed the QSNOW>0 condition so that ISNOW
    # can still be adjusted from SNOWH alone when nothing is falling.
    if col.isnow == 0 and col.snowh >= _f(0.025):
        col.isnow = -1
        newnode = 1
        DZSNSO[0] = col.snowh
        col.snowh = _ZERO
        STC[0] = min(_f(273.16), sfctmp)
        SNICE[0] = col.sneqv
        SNLIQ[0] = _ZERO

    # snow with layers
    if col.isnow < 0 and newnode == 0 and qsnow > _ZERO:
        SNICE[col.isnow + 1] = SNICE[col.isnow + 1] + qsnow * dt
        DZSNSO[col.isnow + 1] = DZSNSO[col.isnow + 1] + snowhin * dt


# ===========================================================================
# COMPACT -- lines 6974-7081
# ===========================================================================

def compact(col: SnowColumn, dt, imelt, ficeold, cbd=None) -> None:
    """Compact the snowpack by metamorphism, overburden and melt.

    ``cbd`` replaces the overburden parameter C2 (``None`` keeps WRF's 0.021,
    the default and byte-identical path); compaction option 2 passes
    :func:`compaction_cbd`.
    """
    dt = _f(dt)
    c2 = _C2 if cbd is None else _f(cbd)
    imelt = np.asarray(imelt, dtype=np.int32)
    ficeold = np.asarray(ficeold, dtype=np.float32)
    IMELT = _Col(imelt, -col.nsnow + 1)
    FICEOLD = _Col(ficeold, -col.nsnow + 1)
    SNICE, SNLIQ, STC, DZSNSO = col.SNICE, col.SNLIQ, col.STC, col.DZSNSO
    fice = _Col(np.zeros(col.nsnow, dtype=np.float32), -col.nsnow + 1)

    burden = _ZERO

    for j in range(col.isnow + 1, 1):
        wx = SNICE[j] + SNLIQ[j]
        fice[j] = SNICE[j] / wx
        void = _ONE - (SNICE[j] / DENICE + SNLIQ[j] / DENH2O) / DZSNSO[j]

        # Allow compaction only for non-saturated node and higher ice lens node.
        if void > _f(0.001) and SNICE[j] > _f(0.1):
            bi = SNICE[j] / DZSNSO[j]
            td = max(_ZERO, TFRZ - STC[j])
            dexpf = _f(expf(-_C4 * td))

            # Settling as a result of destructive metamorphism
            ddz1 = -_C3 * dexpf
            if bi > _DM:
                ddz1 = ddz1 * _f(expf(_f(-46.0e-3) * (bi - _DM)))

            # Liquid water term
            if SNLIQ[j] > _f(0.01) * DZSNSO[j]:
                ddz1 = ddz1 * _C5

            # Compaction due to overburden; 0.5*WX is the self-burden
            ddz2 = (-(burden + _HALF * wx)
                    * _f(expf(_f(-0.08) * td - c2 * bi)) / _ETA0)

            # Compaction occurring during melt
            if IMELT[j] == 1:
                ddz3 = max(_ZERO,
                           (FICEOLD[j] - fice[j]) / max(_f(1.0e-6), FICEOLD[j]))
                ddz3 = -ddz3 / dt
            else:
                ddz3 = _ZERO

            # Time rate of fractional change in DZ (units of s-1)
            pdzdtc = (ddz1 + ddz2 + ddz3) * dt
            pdzdtc = max(_f(-0.5), pdzdtc)

            # The change in DZ due to compaction
            DZSNSO[j] = DZSNSO[j] * (_ONE + pdzdtc)
            DZSNSO[j] = max(DZSNSO[j], SNICE[j] / DENICE + SNLIQ[j] / DENH2O)

            # C.He: constrain snow density to 50~500 kg/m3
            DZSNSO[j] = min(max(DZSNSO[j], (SNICE[j] + SNLIQ[j]) / _f(500.0)),
                            (SNICE[j] + SNLIQ[j]) / _f(50.0))

        # Pressure of overlying snow
        burden = burden + wx


# ===========================================================================
# COMBINE -- lines 6610-6788
# ===========================================================================

def combine(col: SnowColumn, ponding1, ponding2):
    """Merge snow layers that are too thin in mass or in thickness.

    ``ponding1``/``ponding2`` mirror WRF's INTENT(OUT) dummies, which are not
    assigned on every path; pass the caller's current values in and use the
    returned pair.
    """
    ponding1, ponding2 = _f(ponding1), _f(ponding2)
    SNICE, SNLIQ, STC = col.SNICE, col.SNLIQ, col.STC
    DZSNSO, SICE, SH2O = col.DZSNSO, col.SICE, col.SH2O

    isnow_old = col.isnow

    for j in range(isnow_old + 1, 1):
        if SNICE[j] <= _f(0.1):
            if j != 0:
                SNLIQ[j + 1] = SNLIQ[j + 1] + SNLIQ[j]
                SNICE[j + 1] = SNICE[j + 1] + SNICE[j]
                DZSNSO[j + 1] = DZSNSO[j + 1] + DZSNSO[j]
            else:
                if col.isnow < -1:
                    SNLIQ[j - 1] = SNLIQ[j - 1] + SNLIQ[j]
                    SNICE[j - 1] = SNICE[j - 1] + SNICE[j]
                    DZSNSO[j - 1] = DZSNSO[j - 1] + DZSNSO[j]
                else:
                    if SNICE[j] >= _ZERO:
                        ponding1 = SNLIQ[j]
                        col.sneqv = SNICE[j]
                        col.snowh = DZSNSO[j]
                    else:  # SNICE over-sublimated earlier
                        ponding1 = SNLIQ[j] + SNICE[j]
                        if ponding1 < _ZERO:
                            # negative SICE from over-sublimation is fixed below
                            SICE[1] = SICE[1] + ponding1 / (DZSNSO[1] * _f(1000.0))
                            ponding1 = _ZERO
                        col.sneqv = _ZERO
                        col.snowh = _ZERO
                    SNLIQ[j] = _ZERO
                    SNICE[j] = _ZERO
                    DZSNSO[j] = _ZERO

            # shift all elements above this down by one
            if j > col.isnow + 1 and col.isnow < -1:
                for i in range(j, col.isnow + 1, -1):
                    STC[i] = STC[i - 1]
                    SNLIQ[i] = SNLIQ[i - 1]
                    SNICE[i] = SNICE[i - 1]
                    DZSNSO[i] = DZSNSO[i - 1]
            col.isnow = col.isnow + 1

    # to conserve water in case of too large surface sublimation
    if SICE[1] < _ZERO:
        SH2O[1] = SH2O[1] + SICE[1]
        SICE[1] = _ZERO

    if col.isnow == 0:  # MB: get out if no longer multi-layer
        return ponding1, ponding2

    col.sneqv = _ZERO
    col.snowh = _ZERO
    zwice = _ZERO
    zwliq = _ZERO

    for j in range(col.isnow + 1, 1):
        col.sneqv = col.sneqv + SNICE[j] + SNLIQ[j]
        col.snowh = col.snowh + DZSNSO[j]
        zwice = zwice + SNICE[j]
        zwliq = zwliq + SNLIQ[j]

    # check the snow depth - all snow gone; liquid water ponds on the soil
    if col.snowh < _f(0.025) and col.isnow < 0:
        col.isnow = 0
        col.sneqv = zwice
        ponding2 = zwliq
        if col.sneqv <= _ZERO:
            col.snowh = _ZERO

    # check the snow depth - snow layers combined
    if col.isnow < -1:
        isnow_old = col.isnow
        mssi = 1

        for i in range(isnow_old + 1, 1):
            if DZSNSO[i] < _DZMIN[mssi - 1]:
                if i == col.isnow + 1:
                    neibor = i + 1
                elif i == 0:
                    neibor = i - 1
                else:
                    neibor = i + 1
                    if (DZSNSO[i - 1] + DZSNSO[i]) < (DZSNSO[i + 1] + DZSNSO[i]):
                        neibor = i - 1

                # Node l and j are combined and stored as node j.
                if neibor > i:
                    j, l = neibor, i
                else:
                    j, l = i, neibor

                DZSNSO[j], SNLIQ[j], SNICE[j], STC[j] = combo(
                    DZSNSO[j], SNLIQ[j], SNICE[j], STC[j],
                    DZSNSO[l], SNLIQ[l], SNICE[l], STC[l])

                # Now shift all elements above this down one.
                if j - 1 > col.isnow + 1:
                    for k in range(j - 1, col.isnow + 1, -1):
                        STC[k] = STC[k - 1]
                        SNICE[k] = SNICE[k - 1]
                        SNLIQ[k] = SNLIQ[k - 1]
                        DZSNSO[k] = DZSNSO[k - 1]

                # Decrease the number of snow layers
                col.isnow = col.isnow + 1
                if col.isnow >= -1:
                    break
            else:
                # thickness is greater than the prescribed minimum value
                mssi = mssi + 1

    return ponding1, ponding2


# ===========================================================================
# DIVIDE -- lines 6792-6916
# ===========================================================================

def divide(col: SnowColumn) -> None:
    """Subdivide snow layers that have grown past their thickness limits."""
    nsnow = col.nsnow
    STC, SNICE, SNLIQ, DZSNSO = col.STC, col.SNICE, col.SNLIQ, col.DZSNSO

    # WRF leaves the slots above ABS(ISNOW) undefined; poison them so that an
    # accidental read shows up instead of silently reading a zero.
    nan = _f(np.nan)
    dz = _Col(np.full(nsnow, nan, dtype=np.float32), 1)
    swice = _Col(np.full(nsnow, nan, dtype=np.float32), 1)
    swliq = _Col(np.full(nsnow, nan, dtype=np.float32), 1)
    tsno = _Col(np.full(nsnow, nan, dtype=np.float32), 1)

    for j in range(1, nsnow + 1):
        if j <= abs(col.isnow):
            dz[j] = DZSNSO[j + col.isnow]
            swice[j] = SNICE[j + col.isnow]
            swliq[j] = SNLIQ[j + col.isnow]
            tsno[j] = STC[j + col.isnow]

    msno = abs(col.isnow)

    if msno == 1:
        # Specify a new snow layer
        if dz[1] > _f(0.05):
            msno = 2
            dz[1] = dz[1] / _f(2.0)
            swice[1] = swice[1] / _f(2.0)
            swliq[1] = swliq[1] / _f(2.0)
            dz[2] = dz[1]
            swice[2] = swice[1]
            swliq[2] = swliq[1]
            tsno[2] = tsno[1]

    if msno > 1:
        if dz[1] > _f(0.05):
            drr = dz[1] - _f(0.05)
            propor = drr / dz[1]
            zwice = propor * swice[1]
            zwliq = propor * swliq[1]
            propor = _f(0.05) / dz[1]
            swice[1] = propor * swice[1]
            swliq[1] = propor * swliq[1]
            dz[1] = _f(0.05)

            dz[2], swliq[2], swice[2], tsno[2] = combo(
                dz[2], swliq[2], swice[2], tsno[2], drr, zwliq, zwice, tsno[1])

            # subdivide a new layer.  MB raised this limit from 0.10 to 0.20.
            if msno <= 2 and dz[2] > _f(0.20):
                msno = 3
                dtdz = (tsno[1] - tsno[2]) / ((dz[1] + dz[2]) / _f(2.0))
                dz[2] = dz[2] / _f(2.0)
                swice[2] = swice[2] / _f(2.0)
                swliq[2] = swliq[2] / _f(2.0)
                dz[3] = dz[2]
                swice[3] = swice[2]
                swliq[3] = swliq[2]
                tsno[3] = tsno[2] - dtdz * dz[2] / _f(2.0)
                if tsno[3] >= TFRZ:
                    tsno[3] = tsno[2]
                else:
                    tsno[2] = tsno[2] + dtdz * dz[2] / _f(2.0)

    if msno > 2:
        if dz[2] > _f(0.2):
            drr = dz[2] - _f(0.2)
            propor = drr / dz[2]
            zwice = propor * swice[2]
            zwliq = propor * swliq[2]
            propor = _f(0.2) / dz[2]
            swice[2] = propor * swice[2]
            swliq[2] = propor * swliq[2]
            dz[2] = _f(0.2)
            dz[3], swliq[3], swice[3], tsno[3] = combo(
                dz[3], swliq[3], swice[3], tsno[3], drr, zwliq, zwice, tsno[2])

    col.isnow = -msno

    for j in range(col.isnow + 1, 1):
        DZSNSO[j] = dz[j - col.isnow]
        SNICE[j] = swice[j - col.isnow]
        SNLIQ[j] = swliq[j - col.isnow]
        STC[j] = tsno[j - col.isnow]


# ===========================================================================
# SNOWH2O -- lines 7085-7230
# ===========================================================================

def snowh2o(col: SnowColumn, dt, qsnfro, qsnsub, qrain, ssi, snow_ret_fac,
            ponding1, ponding2):
    """Renew snow ice/liquid for sublimation, frost, rain and percolation.

    Returns ``(qsnbot, ponding1, ponding2)``.  The two ponding values are
    only touched by the COMBINE call this routine may make.
    """
    dt, qsnfro, qsnsub, qrain = _f(dt), _f(qsnfro), _f(qsnsub), _f(qrain)
    ssi, snow_ret_fac = _f(ssi), _f(snow_ret_fac)
    ponding1, ponding2 = _f(ponding1), _f(ponding2)

    SNICE, SNLIQ, STC = col.SNICE, col.SNLIQ, col.STC
    DZSNSO, SICE, SH2O = col.DZSNSO, col.SICE, col.SH2O

    vol_liq = _Col(np.zeros(col.nsnow, dtype=np.float32), -col.nsnow + 1)
    vol_ice = _Col(np.zeros(col.nsnow, dtype=np.float32), -col.nsnow + 1)
    epore = _Col(np.zeros(col.nsnow, dtype=np.float32), -col.nsnow + 1)

    # for the case when SNEQV becomes '0' after 'COMBINE'
    if col.sneqv == _ZERO:
        # Barlage: SH2O -> SICE in v3.6
        SICE[1] = SICE[1] + (qsnfro - qsnsub) * dt / (DZSNSO[1] * _f(1000.0))
        if SICE[1] < _ZERO:
            SH2O[1] = SH2O[1] + SICE[1]
            SICE[1] = _ZERO

    # for shallow snow without a layer: excessive sublimation reduces soil water
    if col.isnow == 0 and col.sneqv > _ZERO:
        temp = col.sneqv
        col.sneqv = col.sneqv - qsnsub * dt + qsnfro * dt
        propor = col.sneqv / temp
        col.snowh = max(_ZERO, propor * col.snowh)
        col.snowh = min(max(col.snowh, col.sneqv / _f(500.0)),
                        col.sneqv / _f(50.0))

        if col.sneqv < _ZERO:
            SICE[1] = SICE[1] + col.sneqv / (DZSNSO[1] * _f(1000.0))
            col.sneqv = _ZERO
            col.snowh = _ZERO
        if SICE[1] < _ZERO:
            SH2O[1] = SH2O[1] + SICE[1]
            SICE[1] = _ZERO

    if col.snowh <= _f(1.0e-8) or col.sneqv <= _f(1.0e-6):
        col.snowh = _ZERO
        col.sneqv = _ZERO

    # for deep snow
    if col.isnow < 0:
        wgdif = SNICE[col.isnow + 1] - qsnsub * dt + qsnfro * dt
        SNICE[col.isnow + 1] = wgdif
        if wgdif < _f(1.0e-6) and col.isnow < 0:
            ponding1, ponding2 = combine(col, ponding1, ponding2)
            SNICE, SNLIQ, STC = col.SNICE, col.SNLIQ, col.STC
            DZSNSO, SICE, SH2O = col.DZSNSO, col.SICE, col.SH2O
        # KWM: COMBINE can change ISNOW back to 0
        if col.isnow < 0:
            SNLIQ[col.isnow + 1] = SNLIQ[col.isnow + 1] + qrain * dt
            SNLIQ[col.isnow + 1] = max(_ZERO, SNLIQ[col.isnow + 1])

    # Porosity and partial volume
    for j in range(col.isnow + 1, 1):
        vol_ice[j] = min(_ONE, SNICE[j] / (DZSNSO[j] * DENICE))
        epore[j] = _ONE - vol_ice[j]

    qin = _ZERO
    qout = _ZERO

    for j in range(col.isnow + 1, 1):
        SNLIQ[j] = SNLIQ[j] + qin
        vol_liq[j] = SNLIQ[j] / (DZSNSO[j] * DENH2O)
        qout = max(_ZERO, (vol_liq[j] - ssi * epore[j]) * DZSNSO[j])
        if j == 0:
            qout = max((vol_liq[j] - epore[j]) * DZSNSO[j],
                       snow_ret_fac * dt * qout)
        qout = qout * DENH2O
        SNLIQ[j] = SNLIQ[j] - qout
        if (SNLIQ[j] / (SNICE[j] + SNLIQ[j])) > _MAX_LIQ_MASS_FRACTION:
            qout = qout + (SNLIQ[j] - _MAX_LIQ_MASS_FRACTION
                           / (_ONE - _MAX_LIQ_MASS_FRACTION) * SNICE[j])
            SNLIQ[j] = (_MAX_LIQ_MASS_FRACTION
                        / (_ONE - _MAX_LIQ_MASS_FRACTION) * SNICE[j])
        qin = qout

    for j in range(col.isnow + 1, 1):
        DZSNSO[j] = max(DZSNSO[j], SNLIQ[j] / DENH2O + SNICE[j] / DENICE)

    # Liquid water from snow bottom to soil
    qsnbot = qout / dt

    return qsnbot, ponding1, ponding2


# ===========================================================================
# SNOWWATER -- lines 6398-6535
# ===========================================================================

def snowwater(col: SnowColumn, dt, zsoil, imelt, ficeold, sfctmp, snowhin,
              qsnow, qsnfro, qsnsub, qrain, ssi, snow_ret_fac, *,
              compaction=1, psfc=None):
    """Sequence the snow leaves and rebuild the layer geometry.

    Returns ``(qsnbot, snoflow, ponding1, ponding2)``.  ``compaction=2``
    selects the C_bd(T_air, p_sfc) overburden parameter and needs ``psfc``.
    """
    if compaction not in COMPACTION_OPTIONS:
        raise ValueError(f"snow compaction option {compaction!r} is not 1 or 2")
    if compaction == 2 and psfc is None:
        raise ValueError("snow compaction option 2 needs surface pressure psfc")
    dt = _f(dt)
    zsoil = np.asarray(zsoil, dtype=np.float32)
    ZSOIL = _Col(zsoil, 1)
    nsoil, nsnow = col.nsoil, col.nsnow

    snoflow = _ZERO
    ponding1 = _ZERO
    ponding2 = _ZERO

    snowfall(col, dt, qsnow, snowhin, sfctmp)

    # MB: do each if block separately
    if col.isnow < 0:
        compact(col, dt, imelt, ficeold,
                None if compaction == 1 else compaction_cbd(sfctmp, psfc))
    if col.isnow < 0:
        ponding1, ponding2 = combine(col, ponding1, ponding2)
    if col.isnow < 0:
        divide(col)

    qsnbot, ponding1, ponding2 = snowh2o(
        col, dt, qsnfro, qsnsub, qrain, ssi, snow_ret_fac, ponding1, ponding2)

    SNICE, SNLIQ, STC = col.SNICE, col.SNLIQ, col.STC
    DZSNSO, ZSNSO = col.DZSNSO, col.ZSNSO

    # set empty snow layers to zero
    for iz in range(-nsnow + 1, col.isnow + 1):
        SNICE[iz] = _ZERO
        SNLIQ[iz] = _ZERO
        STC[iz] = _ZERO
        DZSNSO[iz] = _ZERO
        ZSNSO[iz] = _ZERO

    # to obtain equilibrium state of snow in glacier region
    if col.sneqv > _f(5000.0):
        bdsnow = SNICE[0] / DZSNSO[0]
        snoflow = col.sneqv - _f(5000.0)
        SNICE[0] = SNICE[0] - snoflow
        DZSNSO[0] = DZSNSO[0] - snoflow / bdsnow
        snoflow = snoflow / dt

    # sum up snow mass for layered snow
    if col.isnow < 0:  # MB: only do for multi-layer
        col.sneqv = _ZERO
        for iz in range(col.isnow + 1, 1):
            col.sneqv = col.sneqv + SNICE[iz] + SNLIQ[iz]

    # Reset ZSNSO and layer thickness DZSNSO
    for iz in range(col.isnow + 1, 1):
        DZSNSO[iz] = -DZSNSO[iz]

    DZSNSO[1] = ZSOIL[1]
    for iz in range(2, nsoil + 1):
        DZSNSO[iz] = ZSOIL[iz] - ZSOIL[iz - 1]

    ZSNSO[col.isnow + 1] = DZSNSO[col.isnow + 1]
    for iz in range(col.isnow + 2, nsoil + 1):
        ZSNSO[iz] = ZSNSO[iz - 1] + DZSNSO[iz]

    for iz in range(col.isnow + 1, nsoil + 1):
        DZSNSO[iz] = -DZSNSO[iz]

    # C.He: update SNOWH for multi-layer snow
    if col.isnow < 0:
        col.snowh = _ZERO
        for iz in range(col.isnow + 1, 1):
            col.snowh = col.snowh + DZSNSO[iz]

    return qsnbot, snoflow, ponding1, ponding2
