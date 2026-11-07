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
"""WRF v4.6.1 legacy RRTMG option-4 driver-side prep, NumPy FP32 reference.

Port authority: the wrapper subroutines RRTMG_LWRAD (with INIRAD/O3DATA,
RELCALC, REICALC and the module retab table) in phys/module_ra_rrtmg_lw.F
and RRTMG_SWRAD in phys/module_ra_rrtmg_sw.F of the campaign reference
bundle (WRF_1974_MP55_reference_bundle, WRF_source_v4.6.1_group),
compiled FP32 (kind_rb = kind(1.0)), EM_CORE=1, WRF_CHEM=0.  This module
covers everything those wrappers do between their WRF-level dummies and
their rrtmg_lw / rrtmg_sw calls, plus the WRF-level output mapping after
the call:

* :func:`lwrad_prep`    -- RRTMG_LWRAD dummies -> the 31 rrtmg_lw entry args
  (Cavallo 4-hPa buffer layers to nlayers, PPROF/TPROF-anchored tlev/tlay,
  trace-gas profiles, O3DATA climatology + o3input=2 shifted profile,
  CAM cloud water/ice/snow paths with in-cloud division, effective radii
  incl. relcalc/reicalc fallbacks and the 130-um snow discount, McICA via
  woof.core.rrtmg_mcica with permuteseed=150).
* :func:`lwrad_outputs` -- rrtmg_lw fluxes/heating -> GLW, OLR, LWCF,
  LW[UP|DN][T|B][C], RTHRATENLW[C], LW*FLX[C] profiles.
* :func:`swrad_prep`    -- RRTMG_SWRAD dummies -> the rrtmg_sw entry args
  (single extra layer with plev top 1.0e-5 mb -- NOT the Cavallo buffer;
  albedo four-way split, scon = solcon*(1-obscur), adjes=1/dyofyr=0,
  McICA with permuteseed=1).  Refuses night columns (coszen <= 0): WRF
  never calls rrtmg_sw for them.
* :func:`swrad_night_outputs` -- exactly the fields RRTMG_SWRAD writes on
  a night column (COSZR = xcoszen and a fixed list of zeros; GSW,
  RTHRATENSW[C] and the SW*FLX profiles are left UNTOUCHED by WRF).
The SW day-side output mapping already exists as
woof.core.rrtmg_sw.swrad_option4_outputs and is not duplicated here.

Everything is single-column (leading ncol axis dropped; the wrappers
hard-code ncol=1) and bottom-up in the vertical, matching WRF.

Gates: tests/test_rrtmg_legacy_prep.py, bitwise (max_ulp 0) against
oracle fixtures dumped from the UNMODIFIED Fortran
(tools/rrtmg_wrf461_oracle/lw_extract.F90 records the exact rrtmg_lw
arguments under in/* and the wrapper outputs under wrap/*;
sw_fixture_driver.F90 records entry/*, mcin/*, wrf/* and the night flag).

FP32 discipline: identical to woof.core.rrtmg_lw (see its docstring) --
one Fortran statement = the same left-to-right sequence of float32 ops,
F() literals, Fortran INT() = truncation toward zero, NINT() = round half
away from zero, exp on float32 through the audited glibc 2.39 expf.

Driver-side input contract (what the forecast model must supply)
-----------------------------------------------------------------
Per column, bottom-up, all float32:
  p3d (kts:kte) [Pa] layer pressure; p8w (kts:kte+1) [Pa] interface
  pressure (strictly decreasing); t3d (kts:kte) [K]; t8w (kts:kte+1) [K];
  dz8w (kts:kte) [m] layer thickness; pi3d (kts:kte) [-] Exner function
  (needed only by the output mapping); qv/qc/qr/qi/qs/qg 3d (kts:kte)
  [kg/kg] mixing ratios; cldfra3d (kts:kte) [0..1]; re_cloud/re_ice/
  re_snow (kts:kte) [m] effective radii (converted to um here);
  o33d (kts:kte) [mol/mol VMR] when o3input=2 (see units note below).
Per grid point scalars: tsk [K], emiss [-] (LW), albedo [-] (SW),
  xland [1=land, 2=water], xice [0..1], snow [mm water equivalent -- the
  wrapper derives snowh = 0.001*snow itself; the model's SNOWH field is
  NOT used], xlat [deg], xcoszen [-] (SW), solcon [W m-2, already
  eccentricity-adjusted] (SW), obscur [-] eclipse obscuration (SW).
Configuration: icloud, cldovrlp (-> icld), idcor, o3input, has_reqc/i/s,
  warm_rain, f_qc/f_qr/f_qi/f_qs/f_qg, yr, julian [fractional day,
  juldat = INT(julian)], mp_physics (Ferrier guard only), g [m s-2],
  and for LW nlayers from :func:`compute_lw_nlayers` (p_top [Pa]).
rho3d and the WRF ``r`` dummy are accepted by WRF but read only on paths
dead under EM_CORE=1 / option 4 (non-EM ice radii, inflg=0 optics), so
they are deliberately not part of this contract.

o33d units at the wrapper boundary
----------------------------------
o33d is ozone VOLUME mixing ratio (mol/mol) on the model mass layers.
The wrapper performs NO vertical interpolation of o33d: for radiation
layers k <= kte it copies o3vmr(k) = o33d(k) verbatim (those radiation
layers ARE the model layers).  Above the model top it extends the column
with the O3DATA annual-mean climatology *shifted* so it is continuous
with the model's top value:  o3vmr(k) = o33d(kte) - o3mmr(kte)*amdo
+ o3mmr(k)*amdo (amdo = 0.603461 converts the climatology's mass mixing
ratio to VMR), clamped to the unshifted climatology o3mmr(k)*amdo
whenever the shifted value would be <= 0.  o3mmr itself comes from
O3DATA's pressure-thickness-weighted layer average of the 31-level
annual profile (mean of summer and winter-interpolated-to-summer-
pressures), evaluated on the radiation plev.  When o3input /= 2 the
whole column is the unshifted climatology and WRF writes the model-layer
part back into O33D (returned here as ``o31d``).

Fortran oddities found (implemented as WRF defines them, reported):
* RRTMG_LWRAD's buffer-temperature loop runs L = kte+1 .. nlayers+1 and
  therefore OVERWRITES tlev(kte+1) (the model-top interface temperature
  from t8w) and tlay(kte) (the top model layer temperature from t3d)
  with the PPROF/TPROF standard-atmosphere values anchored at
  tlev(kte) - varint(kte) -- the anchor is the interface BELOW the model
  top, not the top interface.  The top model layer's LW heating rate is
  computed from a temperature that is not t3d(kte).
* In the iceflg==5 snow-path loop both wrappers execute
  ``gicewp = gicewp + qs*(1-0.99)*...`` where gicewp is the stale scalar
  left over from the previous DO loop's last iteration (k = kte); the
  result is never read afterwards (cicewp is not updated), so the
  statement is dead code -- the intended 1% snow-into-ice transfer never
  reaches the radiation.  Not modelled (no observable effect).
* The P3 special case (has_reqs=0, has_reqi/=0, has_reqc/=0) sets
  qs1d(k) = QI3D(i,k,j) from the RAW input array, bypassing the
  max(0.,...) clamp applied everywhere else.
* RRTMG_LWRAD derives snowh = 0.001*SNOW for relcalc; the actual SNOWH
  model field is never passed to the radiation wrappers.
* QV3D is copied without consulting F_QV (the flag is accepted and only
  used for presence checks elsewhere).
* relcalc/reicalc (LW) run on tlay AFTER the buffer overwrite above, so
  the k = kte liquid/ice fallback radii use the blended temperature.
* On night columns RRTMG_SWRAD zeroes a fixed list of 2-D diagnostics
  but leaves GSW, RTHRATENSW, RTHRATENSWC and the SW flux profiles
  untouched (stale values from the previous radiation step survive in
  WRF); COSZR is written unconditionally.

Fail-closed exclusions (paths WRF option 4 as configured cannot reach or
that need inputs outside this contract): the HWRF-only Eta Ferrier
package (mp_physics 85), prognostic
aerosol/cloud-drop paths (progn=1, F_QNDROP), WRF_CHEM aerosol feedback,
CAMMGMP radii, the SSiB albedo split (sf_surface_physics == 8), and
o3input values other than the two WRF defines (0-ish default and 2).

Batch twins (column-vectorized, bitwise identical per column)
-------------------------------------------------------------
:func:`lwrad_prep_batch`, :func:`lwrad_outputs_batch`,
:func:`swrad_prep_batch` and :func:`swrad_night_outputs_batch` take the
same logical arguments with a leading ncol axis ((ncol, kte) profiles,
(ncol,) per-grid-point surface scalars) and python-scalar configuration
SHARED by the whole batch (mirroring the batched LW engine's
flag-sharing contract, woof.core.rrtmg_lw Section 11): icloud,
warm_rain, cldovrlp, idcor, o3input, has_req*, f_q*, yr, julian,
nlayers, mp_physics, g, sf_surface_physics and calc_clean_atm_diag are
one value per call -- callers with mixed flags must split the batch by
flag tuple.  Vectorization is across columns only: every elementwise
float32 op of the scalar entries becomes one array op over (ncol, ...)
(NumPy rounds each op identically per element), sequential in-column
accumulations keep their k-loops, and branch ladders become masked
selections with the identical comparison operators, so per-column
results are bitwise equal to the scalar entries at any batch width
(gated in tests/test_rrtmg_legacy_prep.py).

* Day contract: :func:`swrad_prep_batch` takes PRE-GATHERED day columns
  and raises on any coszen <= 0 (the caller gathers day columns, e.g.
  cols = nonzero(xcoszen > 0), and routes the complement through
  :func:`swrad_night_outputs` / :func:`swrad_night_outputs_batch`),
  exactly like WRF's per-column dorrsw gate.
* McICA: the (ncol, ...)-native generators in woof.core.rrtmg_mcica
  are per-column pure (kissvec seeds and every elementwise op act on
  independent column lanes) EXCEPT that ``lat`` is a scalar dummy.  It
  reaches the outputs only through the idcor==1 decorrelation length,
  which is read only by the exponential overlaps (icld 4/5), so outside
  that regime one generator call covers the whole batch; inside it the
  batch entries group columns by identical xlat bit patterns (worst
  case, all-distinct latitudes, this degenerates to per-column calls --
  a real forecast using idcor=1 with cldovrlp 4/5 pays that cost until
  rrtmg_mcica grows a per-column lat dummy).
* The batch entries accept cupy arrays (dispatch on the input array
  namespace); the McICA stage always executes on the host (bit-exact
  scalar RNG chain), staged through bit-preserving float32 copies.
  DAZ/FTZ finding (caught by the cupy gate on the SW fixture deck):
  microphysics mixing ratios traverse the FP32 subnormal range
  (qi3d+qs3d sums at ~1e-39 on real columns), and every kernel cupy
  compiles flushes FP32 subnormal inputs AND results in ALL device
  arithmetic, comparisons and min/max -- only copies and
  where-selects preserve them, and even the f64->f32 cast flushes, so
  FP64 staging cannot materialize a subnormal float32 on the device
  (measured: ciwpth 1.0902803e-33 on numpy vs 0.0 on device).  The
  cause is the compile route rather than the card: cupy appends
  ``-ftz=true`` after the caller's options and the compiler honours
  the last occurrence, which is why ``--ftz=false`` does not lift it
  and why the host-side fallback below is the countermeasure rather
  than a compile flag.  The moisture ->
  effective-radii -> cloud-path block therefore ALWAYS runs on the host
  (numpy-side fallback; its inputs/outputs move as pure data), keeping
  the cupy path bitwise identical to numpy; the remaining device
  arithmetic handles only normal-range level/temperature/ozone/albedo
  values.  This is gated, not assumed.
"""

from __future__ import annotations

import numpy as np

from woof.core.rrtmg_lw import F
from woof.core.rrtmg_mcica import (
    NBNDLW, NBNDSW, LW_PERMUTESEED, SW_PERMUTESEED,
    generate_lw_subcolumns, generate_sw_subcolumns)
from woof.core.rrtmg_sw import option4_trace_gases

__all__ = [
    "compute_lw_nlayers", "lw_trace_gases", "lwrad_prep", "lwrad_outputs",
    "swrad_prep", "swrad_night_outputs",
    "lwrad_prep_batch", "lwrad_outputs_batch", "swrad_prep_batch",
    "swrad_night_outputs_batch",
]

# ---------------------------------------------------------------------------
# Module data (DATA statements of module_ra_rrtmg_lw / RRTMG_LWRAD).
# ---------------------------------------------------------------------------

#: deltap: pressure interval for LW buffer layers (mb).
DELTAP = F("4.0")

_AMDW = F("1.607793")     # molecular weight dry air / water vapor
_AMDO = F("0.603461")     # molecular weight dry air / ozone
_CFC22 = F("0.169e-9")
_CCL4 = F("0.093e-9")
_CFC11 = F("0.251e-9")
_CFC12 = F("0.538e-9")

#: retab(95): tabulated ice effective radius vs temperature (module data).
_RETAB = np.array([
    5.92779, 6.26422, 6.61973, 6.99539, 7.39234,
    7.81177, 8.25496, 8.72323, 9.21800, 9.74075, 10.2930,
    10.8765, 11.4929, 12.1440, 12.8317, 13.5581, 14.2319,
    15.0351, 15.8799, 16.7674, 17.6986, 18.6744, 19.6955,
    20.7623, 21.8757, 23.0364, 24.2452, 25.5034, 26.8125,
    27.7895, 28.6450, 29.4167, 30.1088, 30.7306, 31.2943,
    31.8151, 32.3077, 32.7870, 33.2657, 33.7540, 34.2601,
    34.7892, 35.3442, 35.9255, 36.5316, 37.1602, 37.8078,
    38.4720, 39.1508, 39.8442, 40.5552, 41.2912, 42.0635,
    42.8876, 43.7863, 44.7853, 45.9170, 47.2165, 48.7221,
    50.4710, 52.4980, 54.8315, 57.4898, 60.4785, 63.7898,
    65.5604, 71.2885, 75.4113, 79.7368, 84.2351, 88.8833,
    93.6658, 98.5739, 103.603, 108.752, 114.025, 119.424,
    124.954, 130.630, 136.457, 142.446, 148.608, 154.956,
    161.503, 168.262, 175.248, 182.473, 189.952, 197.699,
    205.728, 214.055, 222.694, 231.661, 240.971, 250.639], dtype=np.float32)

#: PPROF/TPROF(60): weighted-mean standard-atmosphere profiles (Cavallo).
_PPROF = np.array([
    1000.00, 855.47, 731.82, 626.05, 535.57, 458.16,
    391.94, 335.29, 286.83, 245.38, 209.91, 179.57,
    153.62, 131.41, 112.42, 96.17, 82.27, 70.38,
    60.21, 51.51, 44.06, 37.69, 32.25, 27.59,
    23.60, 20.19, 17.27, 14.77, 12.64, 10.81,
    9.25, 7.91, 6.77, 5.79, 4.95, 4.24,
    3.63, 3.10, 2.65, 2.27, 1.94, 1.66,
    1.42, 1.22, 1.04, 0.89, 0.76, 0.65,
    0.56, 0.48, 0.41, 0.35, 0.30, 0.26,
    0.22, 0.19, 0.16, 0.14, 0.12, 0.10], dtype=np.float32)
_TPROF = np.array([
    286.96, 281.07, 275.16, 268.11, 260.56, 253.02,
    245.62, 238.41, 231.57, 225.91, 221.72, 217.79,
    215.06, 212.74, 210.25, 210.16, 210.69, 212.14,
    213.74, 215.37, 216.82, 217.94, 219.03, 220.18,
    221.37, 222.64, 224.16, 225.88, 227.63, 229.51,
    231.50, 233.73, 236.18, 238.78, 241.60, 244.44,
    247.35, 250.33, 253.32, 256.30, 259.22, 262.12,
    264.80, 266.50, 267.59, 268.44, 268.69, 267.76,
    266.13, 263.96, 261.54, 258.93, 256.15, 253.23,
    249.89, 246.67, 243.48, 240.25, 236.66, 233.86], dtype=np.float32)
_NPROFLEVS = 60

# O3DATA tables (31 levels; summer/winter annual blend).
_O3SUM = np.array([
    5.297e-8, 5.852e-8, 6.579e-8, 7.505e-8,
    8.577e-8, 9.895e-8, 1.175e-7, 1.399e-7, 1.677e-7, 2.003e-7,
    2.571e-7, 3.325e-7, 4.438e-7, 6.255e-7, 8.168e-7, 1.036e-6,
    1.366e-6, 1.855e-6, 2.514e-6, 3.240e-6, 4.033e-6, 4.854e-6,
    5.517e-6, 6.089e-6, 6.689e-6, 1.106e-5, 1.462e-5, 1.321e-5,
    9.856e-6, 5.960e-6, 5.960e-6], dtype=np.float32)
_PPSUM = np.array([
    955.890, 850.532, 754.599, 667.742, 589.841,
    519.421, 455.480, 398.085, 347.171, 301.735, 261.310, 225.360,
    193.419, 165.490, 141.032, 120.125, 102.689, 87.829, 75.123,
    64.306, 55.086, 47.209, 40.535, 34.795, 29.865, 19.122,
    9.277, 4.660, 2.421, 1.294, 0.647], dtype=np.float32)
_O3WIN = np.array([
    4.629e-8, 4.686e-8, 5.017e-8, 5.613e-8,
    6.871e-8, 8.751e-8, 1.138e-7, 1.516e-7, 2.161e-7, 3.264e-7,
    4.968e-7, 7.338e-7, 1.017e-6, 1.308e-6, 1.625e-6, 2.011e-6,
    2.516e-6, 3.130e-6, 3.840e-6, 4.703e-6, 5.486e-6, 6.289e-6,
    6.993e-6, 7.494e-6, 8.197e-6, 9.632e-6, 1.113e-5, 1.146e-5,
    9.389e-6, 6.135e-6, 6.135e-6], dtype=np.float32)
_PPWIN = np.array([
    955.747, 841.783, 740.199, 649.538, 568.404,
    495.815, 431.069, 373.464, 322.354, 277.190, 237.635, 203.433,
    174.070, 148.949, 127.408, 108.915, 93.114, 79.551, 67.940,
    58.072, 49.593, 42.318, 36.138, 30.907, 26.362, 16.423,
    7.583, 3.620, 1.807, 0.938, 0.469], dtype=np.float32)


def _build_o3wrk():
    """O3DATA's runtime blend of the summer/winter tables (FP32 op order).

    Executed once: the values depend only on the DATA tables, but every
    arithmetic step keeps the Fortran's single-statement rounding.
    """
    o3ann = np.empty(31, dtype=np.float32)
    o3ann[0] = F(F("0.5") * F(_O3SUM[0] + _O3WIN[0]))
    for k in range(1, 31):     # Fortran K = 2..31
        a = F(_O3WIN[k] - _O3WIN[k - 1])
        b = F(_PPWIN[k] - _PPWIN[k - 1])
        c = F(a / b)
        d = F(_PPSUM[k] - _PPWIN[k - 1])
        o3ann[k] = F(_O3WIN[k - 1] + F(c * d))
    for k in range(1, 31):
        o3ann[k] = F(F("0.5") * F(o3ann[k] + _O3SUM[k]))
    ppwrkh = np.empty(32, dtype=np.float32)
    ppwrkh[0] = F("1100.0")
    for k in range(1, 31):     # Fortran K = 2..31
        ppwrkh[k] = F(F(_PPSUM[k] + _PPSUM[k - 1]) / F("2.0"))
    ppwrkh[31] = F("0.0")
    return o3ann, ppwrkh


_O3WRK, _PPWRKH = _build_o3wrk()

_ZERO = F("0.0")
_HALF = F("0.5")
_ONE = F("1.0")


def _f32(x):
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared building blocks (statement-identical between RRTMG_LWRAD and
# RRTMG_SWRAD).
# ---------------------------------------------------------------------------

def compute_lw_nlayers(kme, p_top, deltap=DELTAP):
    """rrtmg_lwinit: NLAYERS = kme + nint(p_top*0.01/deltap) - 1.

    ``kme`` is the WRF memory top index (kte + 1 for one column),
    ``p_top`` the model top pressure in Pa (FP32 arithmetic, NINT = round
    half away from zero).
    """
    x = F(F(F(p_top) * F("0.01")) / F(deltap))
    xi = float(x)
    n = int(xi)
    if xi - n >= 0.5:
        n += 1
    elif xi - n <= -0.5:
        n -= 1
    # Keep one clear buffer layer above a high model top. Below 2 hPa
    # the rounded 4-hPa extension is zero; using it would remove the
    # buffer and violate the scalar, batched and CUDA prep contracts.
    return int(kme) + max(1, n) - 1


def lw_trace_gases(yr, overrides=None):
    """RRTMG_LWRAD ghg_input=0 trace gases.

    co2/ch4/n2o/o2 shared with the SW helper (REAL(4) exp co2 curve);
    adds the LW-only cfc11/cfc12/cfc22/ccl4 constants.  Returns a dict of
    float32 scalars keyed by the vmr profile names they fill.
    """
    co2, ch4, n2o, o2 = option4_trace_gases(yr)
    gases = {"co2": co2, "ch4": ch4, "n2o": n2o, "o2": o2,
             "cfc11": _CFC11, "cfc12": _CFC12, "cfc22": _CFC22,
             "ccl4": _CCL4}
    if overrides is not None:
        from woof.core.trace_gases import LEGACY_LW_GASES, validate_trace_gas_overrides
        gases.update({name: F(value) for name, value in validate_trace_gas_overrides(
            overrides, supported=LEGACY_LW_GASES, consumer="legacy RRTMG longwave").items()})
    return gases


def _moist_prep(kte, icloud, warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg,
                t1d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, cldfra3d,
                ferrier=False, eta_ferrier=False):
    """The moist-variable assembly common to both wrappers.

    Returns (qv1d, qc1d, qr1d, qi1d, qs1d, qg1d, cldfra1d), each float32
    (kte,).  qg1d mirrors WRF (copied but unused downstream).

    ``ferrier`` (mp_physics 5 or 15) is the wrappers' FER_MP_HIRES arm
    (module_ra_rrtmg_lw.F:12131-12146, module_ra_rrtmg_sw.F:10732-10747):
    OUTSIDE the ICLOUD block, QI1D = MAX(0, QI3D) (the total ice), QS1D = 0
    and QC1D = MAX(0, QC3D).  Under icloud /= 0 it rewrites the same words
    the F_QC/F_QI arms just wrote; under icloud = 0 it hands the optics
    the condensate at zero cloud fraction, as WRF does.
    """
    return _moist_prep_b(
        np, icloud, warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg,
        *map(_f32, (t1d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, cldfra3d)),
        ferrier=ferrier, eta_ferrier=eta_ferrier)


def cloud_form_is_fork(form):
    """Select the public v4.1.21 wrapper cloud rules explicitly.

    SW unsized-liquid radii: module_ra_rrtmg_sw.F:10464-10474.
    Snow path: module_ra_rrtmg_lw.F:12345-12360 and
    module_ra_rrtmg_sw.F:10742-10758. LW liquid radii remain 7.5/10.5.
    """
    if form not in ("wrf_461", "noaa_wrf39"):
        raise ValueError(f"unknown rrtmg_cloud_optics_form={form!r}")
    return form == "noaa_wrf39"


def _effective_radii(kte, icloud, has_reqc, has_reqi, has_reqs,
                     t3d, cldfra3d, xland, re_cloud, re_ice, re_snow,
                     qi3d_raw, qs1d, qi1d, sw_cloud_fallback=False):
    """The has_req* radii blocks + inflg/iceflg resolution + P3 case.

    Returns (inflg, iceflg, recloud1d, reice1d, resnow1d, qs1d, qi1d).
    qs1d/qi1d are replaced (not mutated) when the P3 case fires; note the
    P3 case reads the RAW qi3d (no max(0,.) clamp), as WRF does.
    """
    result = _effective_radii_b(
        np, icloud, has_reqc, has_reqi, has_reqs,
        _f32(t3d)[None], _f32(cldfra3d)[None], np.array([xland], np.float32),
        *(_f32(v)[None] for v in
          (re_cloud, re_ice, re_snow, qi3d_raw, qs1d, qi1d)),
        sw_cloud_fallback=sw_cloud_fallback)
    inflg, iceflg, rec, rei, res, qs, qi = result
    if icloud == 0 or not (has_reqs == 0 and has_reqi != 0 and has_reqc != 0):
        return inflg, iceflg, rec[0], rei[0], res[0], qs1d, qi1d
    return inflg, iceflg, rec[0], rei[0], res[0], qs[0], qi[0]


def _relcalc(t, landfrac, landm, icefrac, snowh):
    """relcalc (Kiehl): liquid effective radius (um) from temperature and
    surface state.  t is (pver,) float32; scalars float32."""
    return _relcalc_b(np, _f32(t)[None], landfrac,
                      np.array([landm], np.float32),
                      np.array([icefrac], np.float32),
                      np.array([snowh], np.float32))[0]


def _reicalc(t):
    """reicalc (Kristjansson & Mitchell): ice effective radius (um) from
    the retab table, index clamped 1..94."""
    return _reicalc_b(np, _f32(t))


def _cloud_properties(kte, iceflg, inflg, qc1d, qi1d, qs1d, cldfra1d,
                      pdel, tlay_model, g, xland, xice, snow,
                      recloud1d, reice1d, resnow1d, cloud_form="wrf_461"):
    """The inflg>0 cloud path/radius block shared by both wrappers.

    ``tlay_model`` is tlay(kts:kte) as seen at the relcalc call site
    (for LW that is AFTER the buffer overwrite of tlay(kte)).
    Returns (clwpth, ciwpth, cswpth, rel, rei, res, resnow1d) over the
    model layers; resnow1d comes back with the 130-um clamp applied.
    """
    result = _cloud_properties_b(
        np, iceflg, inflg,
        *(_f32(v)[None] for v in (qc1d, qi1d, qs1d, cldfra1d, pdel, tlay_model)),
        g, np.array([xland], np.float32), np.array([xice], np.float32),
        np.array([snow], np.float32),
        *(_f32(v)[None] for v in (recloud1d, reice1d, resnow1d)),
        cloud_form=cloud_form)
    return tuple(v[0] for v in result)


def _o3data(prlevh):
    """O3DATA: pressure-thickness-weighted layer average of the annual
    ozone climatology (mass mixing ratio) onto the radiation levels.

    ``prlevh`` are the m+1 radiation interface pressures in mb (bottom
    up, decreasing); returns o3prof, m layer values, float32.
    """
    prlevh = _f32(prlevh)
    m = prlevh.shape[0] - 1
    acc = np.zeros(m, np.float32)
    pk = prlevh[:m]
    pk1 = prlevh[1:m + 1]

    def _neg_part(diff):
        # PB = 0 if -(diff) >= 0 else diff  (i.e. keep only positive diff)
        return np.where(diff > _ZERO, diff, _ZERO).astype(np.float32)

    for jj in range(31):
        pb1 = _neg_part((pk - _PPWRKH[jj]).astype(np.float32))
        pb2 = _neg_part((pk - _PPWRKH[jj + 1]).astype(np.float32))
        pt1 = _neg_part((pk1 - _PPWRKH[jj]).astype(np.float32))
        pt2 = _neg_part((pk1 - _PPWRKH[jj + 1]).astype(np.float32))
        w = (pb2 - pb1).astype(np.float32)
        w = (w - pt2).astype(np.float32)
        w = (w + pt1).astype(np.float32)
        acc = (acc + (w * _O3WRK[jj]).astype(np.float32)).astype(np.float32)
    return (acc / (pk - pk1).astype(np.float32)).astype(np.float32)


def _mp_guard(mp_physics, allowed_extra=()):
    """Refuse the HWRF-only Eta Ferrier package.

    Breakage this prevents: ETAMP_HWRF (mp_physics 85) joins the wrappers'
    Ferrier arm only in an HWRF build (``#if (HWRF == 1)``,
    module_ra_rrtmg_lw.F:12130, _sw.F:10731) and is an NMM-core scheme
    woof does not carry; prepping it as anything would radiate a species
    set the run cannot have.  The ARW arm for mp_physics 5 and 15 is
    ported (``ferrier`` in :func:`_moist_prep`).
    """
    # The historical keyword names additional refused packages.
    # The device shortwave wrapper supplies (85,), never an admission.
    if int(mp_physics) == 85 or int(mp_physics) in tuple(allowed_extra):
        raise NotImplementedError(
            "mp_physics=85 (ETAMP_HWRF) is an HWRF/NMM-only package; its "
            "RRTMG species remap exists only in HWRF builds")


def _is_ferrier(mp_physics):
    """The wrappers' FER_MP_HIRES / FER_MP_HIRES_ADVECT selector test."""
    return int(mp_physics) in (5, 15)


# ---------------------------------------------------------------------------
# Longwave: RRTMG_LWRAD prep and output mapping.
# ---------------------------------------------------------------------------

def _require_radii(entry, has_reqc, has_reqi, has_reqs,
                   re_cloud, re_ice, re_snow):
    """Fail closed on the has_req*/radii contract (prep+ozone audit
    hardening): WRF's has_req* flags declare that the microphysics
    scheme SUPPLIES the corresponding radius field; accepting
    has_req* != 0 with a missing array and silently radiating zeros
    would be the exact silent-wrong the fail-closed rule exists for.
    has_req* == 0 keeps the None -> zeros path (the wrapper then takes
    its relcalc/reicalc fallbacks, which never read the zeros)."""
    for flag, name, arr in ((has_reqc, "re_cloud", re_cloud),
                            (has_reqi, "re_ice", re_ice),
                            (has_reqs, "re_snow", re_snow)):
        if int(flag) != 0 and arr is None:
            raise ValueError(
                f"{entry}: has_{name.replace('re_', 'req')[:4]}"
                f" = {int(flag)} declares the scheme supplies {name}, "
                f"but {name} is None -- supply the field or set the "
                "flag to 0 (fail-closed; zeros would silently corrupt "
                "the cloud optics)")


def lwrad_prep(*, p3d, p8w, t3d, t8w, dz8w,
               qv3d, qc3d=None, qr3d=None, qi3d=None, qs3d=None, qg3d=None,
               cldfra3d=None, o33d=None,
               re_cloud=None, re_ice=None, re_snow=None,
               tsk, emiss, xland, xice, snow, xlat,
               icloud, warm_rain, cldovrlp, idcor, o3input,
               has_reqc, has_reqi, has_reqs,
               f_qc=True, f_qr=True, f_qi=True, f_qs=True, f_qg=True,
               yr, julian, nlayers, mp_physics=0, g=9.81, trace_gas_overrides=None,
                     rrtmg_cloud_optics_form="wrf_461"):
    """RRTMG_LWRAD, from its WRF dummies to the exact rrtmg_lw arguments.

    Single column, bottom-up; layer arrays (kte,), interface arrays
    (kte+1,).  ``nlayers`` comes from :func:`compute_lw_nlayers`.
    Returns a dict keyed by the rrtmg_lw dummy names (the oracle's in/*
    record names): ncol, nlay, icld, play, plev, tlay, tlev, tsfc,
    h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, cfc11vmr, cfc12vmr,
    cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, cldfmcl,
    taucmcl, ciwpmcl, clwpmcl, cswpmcl, reicmcl, relqmcl, resnmcl,
    tauaer; McICA arrays are (ngptlw, nlayers), tauaer (nlayers, nbndlw).
    Extras (not rrtmg_lw args): o31d, hgt, cldfrac, juldat, pdel.
    """
    cloud_form_is_fork(rrtmg_cloud_optics_form)
    _require_radii("lwrad_prep", has_reqc, has_reqi, has_reqs,
                   re_cloud, re_ice, re_snow)
    _mp_guard(mp_physics)
    p3d = _f32(p3d)
    p8w = _f32(p8w)
    kte = p3d.shape[0]
    nlayers = int(nlayers)
    if p8w.shape[0] != kte + 1:
        raise ValueError("p8w must have kte+1 interface values")
    if nlayers < kte + 1:
        raise ValueError("nlayers must be at least kte+1")

    gases = lw_trace_gases(yr, trace_gas_overrides)

    pw1d = (p8w / F("100.0")).astype(np.float32)
    tw1d = _f32(t8w).copy()
    t1d = _f32(t3d).copy()
    p1d = (p3d / F("100.0")).astype(np.float32)
    dz1d = _f32(dz8w).copy()

    if o3input == 2:
        o31d = _f32(o33d).copy()
    else:
        o31d = np.zeros(kte, np.float32)

    zeros = np.zeros(kte, np.float32)
    qv1d, qc1d, qr1d, qi1d, qs1d, qg1d, cldfra1d = _moist_prep(
        kte, icloud, warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg, t1d,
        qv3d,
        zeros if qc3d is None else qc3d, zeros if qr3d is None else qr3d,
        zeros if qi3d is None else qi3d, zeros if qs3d is None else qs3d,
        zeros if qg3d is None else qg3d,
        zeros if cldfra3d is None else cldfra3d,
        ferrier=_is_ferrier(mp_physics), eta_ferrier=(int(mp_physics)==95))

    ncol = 1
    nlay = nlayers
    icld = int(cldovrlp)
    juldat = int(F(julian))          # Fortran INT() on the REAL julian
    liqflglw = 1

    inflglw, iceflglw, recloud1d, reice1d, resnow1d, qs1d, qi1d = \
        _effective_radii(kte, icloud, has_reqc, has_reqi, has_reqs,
                         t3d, zeros if cldfra3d is None else cldfra3d,
                         xland, zeros if re_cloud is None else re_cloud,
                         zeros if re_ice is None else re_ice,
                         zeros if re_snow is None else re_snow,
                         zeros if qi3d is None else qi3d, qs1d, qi1d)

    # ---- level/layer assembly with Cavallo buffer layers ----
    plev = np.zeros(nlayers + 1, np.float32)
    tlev = np.zeros(nlayers + 1, np.float32)
    play = np.zeros(nlayers, np.float32)
    tlay = np.zeros(nlayers, np.float32)
    hgt = np.zeros(nlayers, np.float32)

    plev[0] = pw1d[0]
    tlev[0] = tw1d[0]
    tsfc = F(tsk)
    plev[1:kte + 1] = pw1d[1:kte + 1]
    tlev[1:kte + 1] = tw1d[1:kte + 1]
    play[:kte] = p1d
    tlay[:kte] = t1d
    pdel = (plev[:kte] - plev[1:kte + 1]).astype(np.float32)

    h2ovmr = np.zeros(nlayers, np.float32)
    h2ovmr[:kte] = (qv1d * _AMDW).astype(np.float32)
    co2vmr = np.full(nlayers, gases["co2"], np.float32)
    o2vmr = np.full(nlayers, gases["o2"], np.float32)
    ch4vmr = np.full(nlayers, gases["ch4"], np.float32)
    n2ovmr = np.full(nlayers, gases["n2o"], np.float32)
    cfc11vmr = np.full(nlayers, gases["cfc11"], np.float32)
    cfc12vmr = np.full(nlayers, gases["cfc12"], np.float32)
    cfc22vmr = np.full(nlayers, gases["cfc22"], np.float32)
    ccl4vmr = np.full(nlayers, gases["ccl4"], np.float32)
    # (WRF fills k<=kte then copies layer kte upward; constants make the
    # buffer copies identical to the fill above.)
    h2ovmr[kte:] = h2ovmr[kte - 1]

    dzsum = _ZERO
    for k in range(kte):
        dz = dz1d[k]
        hgt[k] = F(dzsum + F(_HALF * dz))
        dzsum = F(dzsum + dz)

    # Buffer levels above the model top (Cavallo): 4-mb steps.  Same
    # positivity rule as the batched wrapper: the final interface is
    # overwritten with zero, so only n_extra-1 decrements must stay
    # positive. Preserve WRF's 4-mb spacing for ordinary model tops.
    n_extra = nlayers - kte
    dp = DELTAP
    if n_extra > 0 and plev[kte] <= F(DELTAP * F(n_extra - 1)):
        dp = min(DELTAP, F(plev[kte] / F(n_extra)))
    for L in range(kte, nlayers):          # Fortran L = kte+1 .. nlayers
        plev[L + 1] = F(plev[L] - dp)
        play[L] = F(_HALF * F(plev[L] + plev[L + 1]))
        hgt[L] = F(dzsum + F(_HALF * dz))
        dzsum = F(dzsum + dz)
    plev[nlayers] = F("0.00")
    play[nlayers - 1] = F(_HALF * F(plev[nlayers - 1] + plev[nlayers]))

    # Interpolate the PPROF/TPROF standard profile to plev.
    varint = np.zeros(nlayers + 1, np.float32)
    for L in range(nlayers + 1):
        pl = plev[L]
        if float(_PPROF[_NPROFLEVS - 1]) < float(pl):
            klev = None
            for ll in range(2, _NPROFLEVS + 1):
                if float(_PPROF[ll - 1]) < float(pl):
                    klev = ll - 1
                    break
        else:
            klev = _NPROFLEVS
        if klev != _NPROFLEVS:
            vark = _TPROF[klev - 1]
            vark1 = _TPROF[klev]
            wght = F(F(pl - _PPROF[klev - 1])
                     / F(_PPROF[klev] - _PPROF[klev - 1]))
        else:
            vark = _TPROF[klev - 1]
            vark1 = vark
            wght = _ZERO
        varint[L] = F(F(wght * F(vark1 - vark)) + vark)

    # Match the interpolated profile to the WRF column.  NOTE (WRF
    # oddity): the loop starts at L=kte+1 and so overwrites tlev(kte+1)
    # (the model-top interface from t8w) and tlay(kte) (the top model
    # layer from t3d); the anchor is tlev(kte) - varint(kte).
    anchor = F(tlev[kte - 1] - varint[kte - 1])
    for L in range(kte + 1, nlayers + 2):  # Fortran L = kte+1 .. nlayers+1
        tlev[L - 1] = F(varint[L - 1] + anchor)
        tlay[L - 2] = F(_HALF * F(tlev[L - 1] + tlev[L - 2]))

    # Ozone profile including the buffer layers.
    o3mmr = _o3data(plev)                  # nlayers values
    o3vmr = np.zeros(nlayers, np.float32)
    if o3input == 2:
        o3vmr[:kte] = o31d
        shift_base = (o3mmr[kte - 1] * _AMDO).astype(np.float32)
        for k in range(kte, nlayers):
            clim = F(o3mmr[k] * _AMDO)
            v = F(F(o31d[kte - 1] - shift_base) + clim)
            if float(v) <= 0.0:
                v = clim
            o3vmr[k] = v
    else:
        o3vmr = (o3mmr * _AMDO).astype(np.float32)
        o31d = o3vmr[:kte].copy()          # WRF writes this back to O33D

    emis = np.full(NBNDLW, F(emiss), np.float32)

    # ---- cloud properties on model layers (inflglw > 0 always here) ----
    cldfrac = np.zeros(nlayers, np.float32)
    cldfrac[:kte] = cldfra1d
    clwpth = np.zeros(nlayers, np.float32)
    ciwpth = np.zeros(nlayers, np.float32)
    cswpth = np.zeros(nlayers, np.float32)
    rel = np.full(nlayers, F("10.0"), np.float32)
    rei = np.full(nlayers, F("10.0"), np.float32)
    res = np.full(nlayers, F("10.0"), np.float32)
    (clwpth[:kte], ciwpth[:kte], cswpth[:kte], rel[:kte], rei[:kte],
     res[:kte], _resnow) = _cloud_properties(
        kte, iceflglw, inflglw, qc1d, qi1d, qs1d, cldfra1d, pdel,
        tlay[:kte], g, xland, xice, snow, recloud1d, reice1d, resnow1d,
            cloud_form=rrtmg_cloud_optics_form)
    taucld = np.zeros((NBNDLW, nlayers), np.float32)
    # Buffer layers: no clouds (cldfrac 0, radii 10, paths/taucld 0).

    # ---- McICA subcolumn generator (permuteseed = 150) ----
    mc = generate_lw_subcolumns(
        1, ncol, nlay, icld, LW_PERMUTESEED, 0,
        play[None, :], cldfrac[None, :], ciwpth[None, :], clwpth[None, :],
        cswpth[None, :], rei[None, :], rel[None, :], res[None, :],
        taucld[:, None, :], hgt[None, :], int(idcor), juldat, F(xlat))

    tauaer = np.zeros((nlayers, NBNDLW), np.float32)

    return {
        "ncol": ncol, "nlay": nlay, "icld": icld,
        "play": play, "plev": plev, "tlay": tlay, "tlev": tlev,
        "tsfc": tsfc,
        "h2ovmr": h2ovmr, "o3vmr": o3vmr, "co2vmr": co2vmr,
        "ch4vmr": ch4vmr, "n2ovmr": n2ovmr, "o2vmr": o2vmr,
        "cfc11vmr": cfc11vmr, "cfc12vmr": cfc12vmr, "cfc22vmr": cfc22vmr,
        "ccl4vmr": ccl4vmr, "emis": emis,
        "inflglw": inflglw, "iceflglw": iceflglw, "liqflglw": liqflglw,
        "cldfmcl": mc["cldfmcl"][:, 0, :], "taucmcl": mc["taucmcl"][:, 0, :],
        "ciwpmcl": mc["ciwpmcl"][:, 0, :], "clwpmcl": mc["clwpmcl"][:, 0, :],
        "cswpmcl": mc["cswpmcl"][:, 0, :],
        "reicmcl": mc["reicmcl"][0], "relqmcl": mc["relqmcl"][0],
        "resnmcl": mc["resnmcl"][0],
        "tauaer": tauaer,
        # extras (not rrtmg_lw dummies)
        "o31d": o31d, "hgt": hgt, "cldfrac": cldfrac, "juldat": juldat,
        "pdel": pdel,
    }


def lwrad_outputs(*, uflx, dflx, hr, uflxc, dflxc, hrc, pi3d,
                  uflxcln=None, dflxcln=None, calc_clean_atm_diag=0):
    """RRTMG_LWRAD's WRF-level output mapping after the rrtmg_lw call.

    ``uflx``/``dflx``/``uflxc``/``dflxc`` are (nlayers+1,) fluxes in
    W m-2 (bottom-up, level nlayers+1 = 0 mb), ``hr``/``hrc`` (nlayers,)
    heating rates in K/day, ``pi3d`` (kte,) the Exner function on the
    model layers.  Returns GLW, OLR, LWCF, the eight TOA/surface flux
    diagnostics, RTHRATENLW/RTHRATENLWC (K s-1 theta tendency, kte
    layers) and the LW*FLX profiles over kts..kte+2 (WRF's optional
    layer-flux outputs), plus the *CLN quartet when clean-sky fluxes are
    supplied and calc_clean_atm_diag > 0.
    """
    inputs = {name: _f32(value)[None] for name, value in {
        "uflx": uflx, "dflx": dflx, "hr": hr, "uflxc": uflxc,
        "dflxc": dflxc, "hrc": hrc, "pi3d": pi3d}.items()}
    if calc_clean_atm_diag > 0 and uflxcln is not None:
        inputs.update(uflxcln=_f32(uflxcln)[None], dflxcln=_f32(dflxcln)[None])
    return {name: value[0] for name, value in lwrad_outputs_batch(
        **inputs, calc_clean_atm_diag=calc_clean_atm_diag).items()}


# ---------------------------------------------------------------------------
# Shortwave: RRTMG_SWRAD prep and the night contract.
# ---------------------------------------------------------------------------

def _require_spectral_albedo(selector, spectral_albedo):
    # WRF module_ra_rrtmg_sw.F maps ALSWVISDIR/DIF and ALSWNIRDIR/DIF
    # directly to these four engine inputs for SSIBSCHEME.
    if int(selector) == 8 and (spectral_albedo is None or
            set(spectral_albedo) != {"asdir", "asdif", "aldir", "aldif"}):
        raise ValueError("sf_surface_physics=8 requires four spectral albedos: "
                         "asdir, asdif, aldir, aldif")


def swrad_prep(*, p3d, p8w, t3d, t8w, dz8w,
               qv3d, qc3d=None, qr3d=None, qi3d=None, qs3d=None, qg3d=None,
               cldfra3d=None, o33d=None,
               re_cloud=None, re_ice=None, re_snow=None,
               tsk, albedo, xland, xice, snow, xlat,
               xcoszen, solcon, obscur=0.0,
               icloud, warm_rain, cldovrlp, idcor, o3input,
               has_reqc, has_reqi, has_reqs,
               f_qc=True, f_qr=True, f_qi=True, f_qs=True, f_qg=True,
               yr, julian, mp_physics=0, g=9.81,
               sf_surface_physics=2, spectral_albedo=None, trace_gas_overrides=None,
                     rrtmg_cloud_optics_form="wrf_461"):
    """RRTMG_SWRAD, from its WRF dummies to the exact rrtmg_sw arguments.

    Day columns only: raises ValueError when xcoszen <= 0 (WRF's dorrsw
    gate skips the whole prep; see :func:`swrad_night_outputs` for what
    the driver writes instead).  The single extra layer tops at
    plev(kte+2) = 1.0e-5 mb -- the SW wrapper does NOT use the Cavallo
    buffer.  Returns a dict keyed by the rrtmg_sw dummy names (the
    oracle's entry/* records): play, plev, tlay, tlev, tsfc, h2ovmr,
    o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, asdir, asdif, aldir, aldif,
    coszen, adjes, dyofyr, scon, icld, inflgsw, iceflgsw, liqflgsw,
    nlay, cldfmcl, taucmcl, ssacmcl, asmcmcl, fsfcmcl, ciwpmcl, clwpmcl,
    cswpmcl, reicmcl, relqmcl, resnmcl -- McICA arrays (ngptsw, nlay) --
    plus coszr (the WRF 2-D diagnostic, = xcoszen) and mcica_inputs
    (the exact mcica_subcol_sw call record for gating).
    """
    fork_cloud = cloud_form_is_fork(rrtmg_cloud_optics_form)
    _require_radii("swrad_prep", has_reqc, has_reqi, has_reqs,
                   re_cloud, re_ice, re_snow)
    _mp_guard(mp_physics)
    _require_spectral_albedo(sf_surface_physics, spectral_albedo)

    coszr = F(xcoszen)
    coszrs = F(xcoszen)
    if float(coszrs) <= 0.0:
        raise ValueError(
            "night column (coszen <= 0): WRF's dorrsw gate skips rrtmg_sw "
            "entirely; use swrad_night_outputs for the driver-side writes")

    p3d = _f32(p3d)
    p8w = _f32(p8w)
    kte = p3d.shape[0]
    if p8w.shape[0] != kte + 1:
        raise ValueError("p8w must have kte+1 interface values")

    co2, ch4, n2o, o2 = option4_trace_gases(yr, trace_gas_overrides)

    pw1d = (p8w / F("100.0")).astype(np.float32)
    tw1d = _f32(t8w).copy()
    t1d = _f32(t3d).copy()
    p1d = (p3d / F("100.0")).astype(np.float32)
    dz1d = _f32(dz8w).copy()

    if o3input == 2:
        o31d = _f32(o33d).copy()
    else:
        o31d = np.zeros(kte, np.float32)

    zeros = np.zeros(kte, np.float32)
    qv1d, qc1d, qr1d, qi1d, qs1d, qg1d, cldfra1d = _moist_prep(
        kte, icloud, warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg, t1d,
        qv3d,
        zeros if qc3d is None else qc3d, zeros if qr3d is None else qr3d,
        zeros if qi3d is None else qi3d, zeros if qs3d is None else qs3d,
        zeros if qg3d is None else qg3d,
        zeros if cldfra3d is None else cldfra3d,
        ferrier=_is_ferrier(mp_physics), eta_ferrier=(int(mp_physics)==95))

    ncol = 1
    nlay = kte + 1
    icld = int(cldovrlp)
    juldat = int(F(julian))
    liqflgsw = 1

    inflgsw, iceflgsw, recloud1d, reice1d, resnow1d, qs1d, qi1d = \
        _effective_radii(kte, icloud, has_reqc, has_reqi, has_reqs,
                         t3d, zeros if cldfra3d is None else cldfra3d,
                         xland, zeros if re_cloud is None else re_cloud,
                         zeros if re_ice is None else re_ice,
                         zeros if re_snow is None else re_snow,
                         zeros if qi3d is None else qi3d, qs1d, qi1d, sw_cloud_fallback=fork_cloud)

    coszen = coszrs
    scon = F(F(solcon) * F(_ONE - F(obscur)))
    dyofyr = 0                 # WRF: solcon already carries eccentricity
    adjes = _ONE

    # ---- level/layer assembly with ONE extra layer ----
    plev = np.zeros(nlay + 1, np.float32)
    tlev = np.zeros(nlay + 1, np.float32)
    play = np.zeros(nlay, np.float32)
    tlay = np.zeros(nlay, np.float32)
    hgt = np.zeros(nlay, np.float32)

    plev[0] = pw1d[0]
    tlev[0] = tw1d[0]
    tsfc = F(tsk)
    plev[1:kte + 1] = pw1d[1:kte + 1]
    tlev[1:kte + 1] = tw1d[1:kte + 1]
    play[:kte] = p1d
    tlay[:kte] = t1d
    pdel = (plev[:kte] - plev[1:kte + 1]).astype(np.float32)

    h2ovmr = np.zeros(nlay, np.float32)
    h2ovmr[:kte] = (qv1d * _AMDW).astype(np.float32)
    co2vmr = np.full(nlay, co2, np.float32)
    o2vmr = np.full(nlay, o2, np.float32)
    ch4vmr = np.full(nlay, ch4, np.float32)
    n2ovmr = np.full(nlay, n2o, np.float32)

    dzsum = _ZERO
    for k in range(kte):
        dz = dz1d[k]
        hgt[k] = F(dzsum + F(_HALF * dz))
        dzsum = F(dzsum + dz)

    play[kte] = F(_HALF * plev[kte])
    tlay[kte] = F(tlev[kte] + _ZERO)
    plev[kte + 1] = F("1.0e-5")
    tlev[kte + 1] = F(tlev[kte] + _ZERO)
    h2ovmr[kte] = h2ovmr[kte - 1]
    hgt[kte] = F(dzsum + F(_HALF * dz))

    o3mmr = _o3data(plev)                  # kte+1 layer values
    o3vmr = np.zeros(nlay, np.float32)
    if o3input == 2:
        o3vmr[:kte] = o31d
        clim = F(o3mmr[kte] * _AMDO)
        v = F(F(o31d[kte - 1] - F(o3mmr[kte - 1] * _AMDO)) + clim)
        if float(v) <= 0.0:
            v = clim
        o3vmr[kte] = v
    else:
        o3vmr = (o3mmr * _AMDO).astype(np.float32)
        o31d = o3vmr[:kte].copy()

    # Albedo split (non-SSiB branch): all four components identical.
    asdir = F(albedo if spectral_albedo is None else spectral_albedo["asdir"])
    asdif = F(albedo if spectral_albedo is None else spectral_albedo["asdif"])
    aldir = F(albedo if spectral_albedo is None else spectral_albedo["aldir"])
    aldif = F(albedo if spectral_albedo is None else spectral_albedo["aldif"])

    # ---- cloud properties on model layers (inflgsw > 0 always here) ----
    cldfrac = np.zeros(nlay, np.float32)
    cldfrac[:kte] = cldfra1d
    clwpth = np.zeros(nlay, np.float32)
    ciwpth = np.zeros(nlay, np.float32)
    cswpth = np.zeros(nlay, np.float32)
    rel = np.full(nlay, F("10.0"), np.float32)
    rei = np.full(nlay, F("10.0"), np.float32)
    res = np.full(nlay, F("10.0"), np.float32)
    (clwpth[:kte], ciwpth[:kte], cswpth[:kte], rel[:kte], rei[:kte],
     res[:kte], _resnow) = _cloud_properties(
        kte, iceflgsw, inflgsw, qc1d, qi1d, qs1d, cldfra1d, pdel,
        tlay[:kte], g, xland, xice, snow, recloud1d, reice1d, resnow1d,
            cloud_form=rrtmg_cloud_optics_form)
    taucld = np.zeros((NBNDSW, nlay), np.float32)
    ssacld = np.ones((NBNDSW, nlay), np.float32)
    asmcld = np.zeros((NBNDSW, nlay), np.float32)
    fsfcld = np.zeros((NBNDSW, nlay), np.float32)
    # Extra layer: no cloud (zeros/tens above set it already).

    mcica_inputs = {
        "play": play.copy(), "cldfrac": cldfrac.copy(),
        "ciwpth": ciwpth.copy(), "clwpth": clwpth.copy(),
        "cswpth": cswpth.copy(), "rei": rei.copy(), "rel": rel.copy(),
        "res": res.copy(), "hgt": hgt.copy(), "icld": icld,
        "idcor": int(idcor), "juldat": juldat, "lat": F(xlat),
        "permuteseed": SW_PERMUTESEED, "irng": 0,
    }

    mc = generate_sw_subcolumns(
        1, ncol, nlay, icld, SW_PERMUTESEED, 0,
        play[None, :], cldfrac[None, :], ciwpth[None, :], clwpth[None, :],
        cswpth[None, :], rei[None, :], rel[None, :], res[None, :],
        taucld[:, None, :], ssacld[:, None, :], asmcld[:, None, :],
        fsfcld[:, None, :], hgt[None, :], int(idcor), juldat, F(xlat))

    return {
        "ncol": ncol, "nlay": nlay, "icld": icld,
        "play": play, "plev": plev, "tlay": tlay, "tlev": tlev,
        "tsfc": tsfc,
        "h2ovmr": h2ovmr, "o3vmr": o3vmr, "co2vmr": co2vmr,
        "ch4vmr": ch4vmr, "n2ovmr": n2ovmr, "o2vmr": o2vmr,
        "asdir": asdir, "asdif": asdif, "aldir": aldir, "aldif": aldif,
        "coszen": coszen, "adjes": adjes, "dyofyr": dyofyr, "scon": scon,
        "inflgsw": inflgsw, "iceflgsw": iceflgsw, "liqflgsw": liqflgsw,
        "cldfmcl": mc["cldfmcl"][:, 0, :], "taucmcl": mc["taucmcl"][:, 0, :],
        "ssacmcl": mc["ssacmcl"][:, 0, :], "asmcmcl": mc["asmcmcl"][:, 0, :],
        "fsfcmcl": mc["fsfcmcl"][:, 0, :], "ciwpmcl": mc["ciwpmcl"][:, 0, :],
        "clwpmcl": mc["clwpmcl"][:, 0, :], "cswpmcl": mc["cswpmcl"][:, 0, :],
        "reicmcl": mc["reicmcl"][0], "relqmcl": mc["relqmcl"][0],
        "resnmcl": mc["resnmcl"][0],
        # WRF-level extras
        "coszr": coszr, "o31d": o31d, "juldat": juldat, "pdel": pdel,
        "mcica_inputs": mcica_inputs,
    }


#: 2-D diagnostics RRTMG_SWRAD explicitly zeroes on a night column.
SW_NIGHT_ZEROED = (
    "swupt", "swuptc", "swdnt", "swdntc",
    "swupb", "swupbc", "swdnb", "swdnbc",
    "swvisdir", "swvisdif", "swnirdir", "swnirdif",
    "swddir", "swddni", "swddif", "swdownc", "swddnic", "swddirc",
    "swcf",
)

#: Outputs RRTMG_SWRAD leaves UNTOUCHED on a night column (stale values
#: survive in WRF; the forecast model must zero or carry them itself).
SW_NIGHT_UNTOUCHED = (
    "gsw", "rthratensw", "rthratenswc",
    "swupflx", "swupflxc", "swdnflx", "swdnflxc",
)


def swrad_night_outputs(xcoszen, calc_clean_atm_diag=0):
    """Exactly what RRTMG_SWRAD writes for a column with coszen <= 0.

    COSZR is set to xcoszen (it is written before the dorrsw gate), the
    :data:`SW_NIGHT_ZEROED` diagnostics are set to 0, and -- deliberately
    absent here -- GSW, RTHRATENSW, RTHRATENSWC and the SW flux profiles
    are left untouched by WRF (:data:`SW_NIGHT_UNTOUCHED`).
    """
    out = {"coszr": F(xcoszen)}
    for name in SW_NIGHT_ZEROED:
        out[name] = _ZERO
    if calc_clean_atm_diag > 0:
        for name in ("swuptcln", "swdntcln", "swupbcln", "swdnbcln"):
            out[name] = _ZERO
    return out


# ---------------------------------------------------------------------------
# Batch twins: column-vectorized entries, bitwise identical per column to
# the scalar entries above (see the module docstring's batch section for
# the contract).  Vectorization is across columns ONLY; every statement
# keeps the scalar code's single-rounding float32 op sequence, sequential
# in-column accumulations keep their k-loops, and branch ladders become
# masked selections with the identical comparison operators (mutually
# exclusive masks, so ladder order is preserved).  ``xp`` is the array
# namespace (numpy, or cupy when the inputs are cupy arrays).
# ---------------------------------------------------------------------------

def _get_xp(*arrays):
    """numpy, or cupy when any input is a cupy array (no cupy import
    unless one actually shows up)."""
    for a in arrays:
        mod = type(a).__module__
        if mod is not None and mod.split(".")[0] == "cupy":
            import cupy
            return cupy
    return np


_XP_TABLES = {}


def _tables(xp):
    """Module DATA tables in the target namespace (device copies of
    float32 constants are bit-preserving; -_PPROF negation is exact)."""
    key = (xp.__name__, int(xp.cuda.Device().id),
           int(xp.cuda.get_current_stream().ptr)) if hasattr(xp, "cuda") else xp.__name__
    if key not in _XP_TABLES:
        _XP_TABLES[key] = {
            "retab": xp.asarray(_RETAB),
            "pprof": xp.asarray(_PPROF),
            "neg_pprof": xp.asarray(-_PPROF),
            "tprof": xp.asarray(_TPROF),
            "o3wrk": xp.asarray(_O3WRK),
            "ppwrkh": xp.asarray(_PPWRKH),
        }
    return _XP_TABLES[key]


def _trunc_int_b(xp, x):
    """Fortran INT() on an already-float32 array: trunc, then int32
    (the same chain as :func:`woof.core.rrtmg_lw.trunc_int`)."""
    return xp.trunc(x).astype(np.int32)


def _host_f32(a):
    """Bit-preserving host view/copy of a float32 array (cupy -> numpy
    device-to-host transfer moves bytes, never rounds)."""
    if isinstance(a, np.ndarray):
        return a
    return a.get()


def _prof_b(xp, v, ncol, nk, name):
    a = xp.asarray(v, np.float32)
    if a.shape != (ncol, nk):
        raise ValueError(
            f"{name} must have shape ({ncol}, {nk}); got {a.shape}")
    return a


def _surf_b(xp, v, ncol, name):
    a = xp.asarray(v, np.float32)
    if a.ndim == 0:
        return xp.broadcast_to(a, (ncol,)).astype(np.float32)
    if a.shape != (ncol,):
        raise ValueError(
            f"{name} must be a scalar or shape ({ncol},); got {a.shape}")
    return a


def _moist_prep_b(xp, icloud, warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg,
                  t1d, qv3d, qc3d, qr3d, qi3d, qs3d, qg3d, cldfra3d,
                  ferrier=False, eta_ferrier=False):
    """Batch twin of :func:`_moist_prep` on (ncol, kte) float32 arrays."""
    f32 = np.float32
    shape = t1d.shape
    qc1d = xp.zeros(shape, f32)
    qr1d = xp.zeros(shape, f32)
    qi1d = xp.zeros(shape, f32)
    qs1d = xp.zeros(shape, f32)
    qg1d = xp.zeros(shape, f32)
    cldfra1d = xp.zeros(shape, f32)

    qv1d = xp.maximum(_ZERO, qv3d).astype(f32)

    if icloud != 0:
        cldfra1d = cldfra3d.copy()
        if f_qc:
            qc1d = xp.maximum(_ZERO, qc3d).astype(f32)
        if f_qr:
            qr1d = xp.maximum(_ZERO, qr3d).astype(f32)
        if (not f_qi) and (not warm_rain):
            cold = t1d < F("273.15")
            qi1d = xp.where(cold, qc1d, qi1d).astype(f32)
            qs1d = xp.where(cold, qr1d, qs1d).astype(f32)
            qc1d = xp.where(cold, _ZERO, qc1d).astype(f32)
            qr1d = xp.where(cold, _ZERO, qr1d).astype(f32)
        if f_qi:
            qi1d = xp.maximum(_ZERO, qi3d).astype(f32)
        if f_qs:
            qs1d = xp.maximum(_ZERO, qs3d).astype(f32)
        if f_qg:
            qg1d = xp.maximum(_ZERO, qg3d).astype(f32)
        if eta_ferrier:
            qi1d = xp.maximum(_ZERO, F("0.1") * qs3d).astype(f32)
            qs1d = (F("0.9") * qs3d).astype(f32)
            qc1d = xp.maximum(_ZERO, qc3d).astype(f32)

    if ferrier:                      # see _moist_prep
        qi1d = xp.maximum(_ZERO, qi3d).astype(f32)
        qs1d = xp.zeros(shape, f32)
        qc1d = xp.maximum(_ZERO, qc3d).astype(f32)

    qv1d = xp.maximum(qv1d, F("1.e-12")).astype(f32)
    return qv1d, qc1d, qr1d, qi1d, qs1d, qg1d, cldfra1d


def _effective_radii_b(xp, icloud, has_reqc, has_reqi, has_reqs,
                       t3d, cldfra3d, xland, re_cloud, re_ice, re_snow,
                       qi3d_raw, qs1d, qi1d, sw_cloud_fallback=False):
    """Batch twin of :func:`_effective_radii`; xland is (ncol,), and the
    scalar ocean/land if-ladder becomes two mutually exclusive masked
    selections with the identical F32 comparison against 1.5."""
    f32 = np.float32
    shape = qs1d.shape
    inflg = 2
    iceflg = 3
    if icloud == 0:
        return (inflg, iceflg, xp.zeros(shape, f32), xp.zeros(shape, f32),
                xp.zeros(shape, f32), qs1d, qi1d)

    xdiff = (xland - F("1.5")).astype(f32)[:, None]

    if has_reqc != 0:
        inflg = 3
        rec = xp.maximum(F("2.5"), (re_cloud * F("1.e6")).astype(
            f32)).astype(f32)
        cloudy = cldfra3d > _ZERO
        rec = xp.where((xdiff > _ZERO) & (rec <= F("2.5")) & cloudy,
                       F("9.6" if sw_cloud_fallback else "10.5"), rec).astype(f32)
        rec = xp.where((xdiff < _ZERO) & (rec <= F("2.5")) & cloudy,
                       F("5.4" if sw_cloud_fallback else "7.5"), rec).astype(f32)
        recloud1d = rec
    else:
        recloud1d = xp.full(shape, F("5.0"), f32)

    if has_reqi != 0:
        inflg = 4
        iceflg = 4
        rei = xp.maximum(F("5.0"), (re_ice * F("1.e6")).astype(
            f32)).astype(f32)
        redo = (rei <= F("5.0")) & (cldfra3d > _ZERO)
        if bool(redo.any()):
            retab = _tables(xp)["retab"]
            idx = _trunc_int_b(xp, (t3d - F("179.0")).astype(f32))
            idx = xp.minimum(xp.maximum(idx, 1), 75)
            corr = (t3d - _trunc_int_b(xp, t3d).astype(f32)).astype(f32)
            interp = ((retab[idx - 1] * (_ONE - corr).astype(f32)
                       ).astype(f32)
                      + (retab[idx] * corr).astype(f32)).astype(f32)
            interp = xp.maximum(interp, F("5.0")).astype(f32)
            rei = xp.where(redo, interp, rei).astype(f32)
        reice1d = rei
    else:
        reice1d = xp.full(shape, F("10.0"), f32)   # EM_CORE==1

    if has_reqs != 0:
        inflg = 5
        iceflg = 5
        resnow1d = xp.maximum(F("10.0"), (re_snow * F("1.e6")).astype(
            f32)).astype(f32)
    else:
        resnow1d = xp.full(shape, F("10.0"), f32)

    if has_reqs == 0 and has_reqi != 0 and has_reqc != 0:
        inflg = 5
        iceflg = 5
        resnow1d = xp.maximum(F("10.0"), (re_ice * F("1.e6")).astype(
            f32)).astype(f32)
        qs1d = qi3d_raw.copy()    # RAW field, no max(0,.) -- WRF oddity
        qi1d = xp.zeros(shape, f32)
        reice1d = xp.full(shape, F("10.0"), f32)

    return inflg, iceflg, recloud1d, reice1d, resnow1d, qs1d, qi1d


def _relcalc_b(xp, t, landfrac, landm, icefrac, snowh):
    """Batch twin of :func:`_relcalc`; t is (ncol, pver), the surface
    scalars (ncol,) broadcast per column."""
    f32 = np.float32
    tmelt = F("273.16")
    rliqocean = F("14.0")
    rliqice = F("14.0")
    rliqland = F("8.0")
    del landfrac  # accepted like the Fortran dummy; landm carries it
    w = ((tmelt - t).astype(f32) * F("0.05")).astype(f32)
    w = xp.minimum(_ONE, xp.maximum(_ZERO, w)).astype(f32)
    rel = (rliqland + (F(rliqocean - rliqland) * w).astype(f32)).astype(f32)
    w = xp.minimum(_ONE, xp.maximum(_ZERO, (snowh * F("10.0")).astype(
        f32))).astype(f32)[:, None]
    rel = (rel + ((rliqocean - rel).astype(f32) * w).astype(f32)).astype(f32)
    w = xp.minimum(_ONE, xp.maximum(_ZERO, (_ONE - landm).astype(
        f32))).astype(f32)[:, None]
    rel = (rel + ((rliqocean - rel).astype(f32) * w).astype(f32)).astype(f32)
    w = xp.minimum(_ONE, xp.maximum(_ZERO, icefrac)).astype(f32)[:, None]
    rel = (rel + ((rliqice - rel).astype(f32) * w).astype(f32)).astype(f32)
    return rel


def _reicalc_b(xp, t):
    """Batch twin of :func:`_reicalc` on (ncol, pver)."""
    f32 = np.float32
    retab = _tables(xp)["retab"]
    idx = _trunc_int_b(xp, (t - F("179.0")).astype(f32))
    idx = xp.minimum(xp.maximum(idx, 1), 94)
    corr = (t - _trunc_int_b(xp, t).astype(f32)).astype(f32)
    return ((retab[idx - 1] * (_ONE - corr).astype(f32)).astype(f32)
            + (retab[idx] * corr).astype(f32)).astype(f32)


def _cloud_properties_b(xp, iceflg, inflg, qc1d, qi1d, qs1d, cldfra1d,
                        pdel, tlay_model, g, xland, xice, snow,
                        recloud1d, reice1d, resnow1d, cloud_form="wrf_461"):
    """Batch twin of :func:`_cloud_properties` on (ncol, kte) arrays with
    (ncol,) surface scalars."""
    f32 = np.float32
    shape = qc1d.shape
    gravmks = F(g)
    landfrac = (F("2.0") - xland).astype(f32)
    landm = landfrac
    snowh = (F("0.001") * snow).astype(f32)
    icefrac = xice

    cldden = xp.maximum(F("0.01"), cldfra1d).astype(f32)

    def _path(q):
        a = (q * pdel).astype(f32)
        b = (a * F("100.0")).astype(f32)
        c = (b / gravmks).astype(f32)
        return (c * F("1000.0")).astype(f32)

    gicewp = _path((qi1d + qs1d).astype(f32))
    gliqwp = _path(qc1d)
    cicewp = (gicewp / cldden).astype(f32)
    cliqwp = (gliqwp / cldden).astype(f32)

    if iceflg >= 4:
        gicewp = _path(qi1d)
        cicewp = (gicewp / cldden).astype(f32)

    csnowp = xp.zeros(shape, f32)
    resnow1d = resnow1d.copy()
    if iceflg == 5:
        smf = xp.full(shape, F("1.0" if cloud_form_is_fork(cloud_form) else "0.99"), f32)
        big = resnow1d > F("130.0")
        q = (F("130.0") / resnow1d).astype(f32)
        smf = xp.where(big, xp.minimum(smf, (q * q).astype(f32)),
                       smf).astype(f32)
        resnow1d = xp.where(big, F("130.0"), resnow1d).astype(f32)
        gsnowp = _path((qs1d * smf).astype(f32))
        csnowp = (gsnowp / cldden).astype(f32)

    reliq = _relcalc_b(xp, tlay_model, landfrac, landm, icefrac, snowh)
    reice = _reicalc_b(xp, tlay_model)

    if inflg >= 3:
        reliq = recloud1d.copy()
    if iceflg >= 4:
        reice = reice1d.copy()
    if iceflg == 3:
        reice = (reice * F("1.0315")).astype(f32)
        reice = xp.minimum(F("140.0"), reice).astype(f32)

    clwpth = cliqwp.copy()
    ciwpth = cicewp.copy()
    rel = reliq.copy()
    rei = reice.copy()
    if inflg == 5:
        cswpth = csnowp.copy()
        res = resnow1d.copy()
    else:
        cswpth = xp.zeros(shape, f32)
        res = xp.full(shape, F("10.0"), f32)
    return clwpth, ciwpth, cswpth, rel, rei, res, resnow1d


_PERFWAVE_DEVICE_XP = "unset"


def _perfwave_device_xp():
    """CuPy when a usable CUDA device is present, else None.  Cached.

    Import failure or a device-less box falls back to the host path, so
    this is a performance decision only -- never a correctness one.
    """
    global _PERFWAVE_DEVICE_XP
    if _PERFWAVE_DEVICE_XP == "unset":
        module = None
        try:
            import cupy as _cupy

            if int(_cupy.cuda.runtime.getDeviceCount()) > 0:
                module = _cupy
        except Exception:
            module = None
        _PERFWAVE_DEVICE_XP = module
    return _PERFWAVE_DEVICE_XP


def _o3data_b_device(cp, prlevh):
    """Device twin of :func:`_o3data_b` -- same float32 ops, same order.

    Every element is still ``(pk - ppwrkh[jj])``, ``_neg_part`` of it,
    ``(pb2-pb1)-pt2+pt1``, ``w*o3wrk[jj]`` in float32, computed exactly
    once.  The only structural change is that the 31 table intervals are
    evaluated as one batched elementwise expression over a leading jj
    axis instead of 31 separate ones; elementwise float32 ops are
    per-element independent, so this cannot move a bit.  The
    accumulation over jj stays STRICTLY SEQUENTIAL, so the float32
    rounding chain of ``acc`` -- the only order-sensitive part -- is
    untouched.  ``astype(f32, copy=False)`` replaces ``astype(f32)``
    where the operand is already float32: a float32->float32 cast is the
    identity, so this drops a copy kernel, not a rounding.
    """
    f32 = np.float32
    tabs = _tables(cp)
    o3wrk = tabs["o3wrk"]
    ppwrkh = tabs["ppwrkh"]
    m = prlevh.shape[1] - 1
    pk = prlevh[:, :m]
    pk1 = prlevh[:, 1:m + 1]

    def _neg_part(diff):
        return cp.where(diff > _ZERO, diff, _ZERO).astype(f32, copy=False)

    lower = ppwrkh[0:31][:, None, None]
    upper = ppwrkh[1:32][:, None, None]
    pk_b = pk[None, :, :]
    pk1_b = pk1[None, :, :]

    pb1 = _neg_part((pk_b - lower).astype(f32, copy=False))
    pb2 = _neg_part((pk_b - upper).astype(f32, copy=False))
    pt1 = _neg_part((pk1_b - lower).astype(f32, copy=False))
    pt2 = _neg_part((pk1_b - upper).astype(f32, copy=False))
    w = (pb2 - pb1).astype(f32, copy=False)
    del pb1, pb2
    w = (w - pt2).astype(f32, copy=False)
    del pt2
    w = (w + pt1).astype(f32, copy=False)
    del pt1
    contribution = (w * o3wrk[:, None, None]).astype(f32, copy=False)
    del w

    acc = cp.zeros((prlevh.shape[0], m), f32)
    for jj in range(31):
        acc = (acc + contribution[jj]).astype(f32, copy=False)
    return (acc / (pk - pk1).astype(f32, copy=False)).astype(f32, copy=False)


def _o3data_b(xp, prlevh):
    """Batch twin of :func:`_o3data`; prlevh is (ncol, m+1)."""
    if xp is np:
        device = _perfwave_device_xp()
        if device is not None:
            return device.asnumpy(
                _o3data_b_device(device, device.asarray(prlevh))
            )
    f32 = np.float32
    tabs = _tables(xp)
    o3wrk = tabs["o3wrk"]
    ppwrkh = tabs["ppwrkh"]
    m = prlevh.shape[1] - 1
    acc = xp.zeros((prlevh.shape[0], m), f32)
    pk = prlevh[:, :m]
    pk1 = prlevh[:, 1:m + 1]

    def _neg_part(diff):
        return xp.where(diff > _ZERO, diff, _ZERO).astype(f32)

    for jj in range(31):
        pb1 = _neg_part((pk - ppwrkh[jj]).astype(f32))
        pb2 = _neg_part((pk - ppwrkh[jj + 1]).astype(f32))
        pt1 = _neg_part((pk1 - ppwrkh[jj]).astype(f32))
        pt2 = _neg_part((pk1 - ppwrkh[jj + 1]).astype(f32))
        w = (pb2 - pb1).astype(f32)
        w = (w - pt2).astype(f32)
        w = (w + pt1).astype(f32)
        acc = (acc + (w * o3wrk[jj]).astype(f32)).astype(f32)
    return (acc / (pk - pk1).astype(f32)).astype(f32)


def _varint_b(xp, plev):
    """Batch twin of the PPROF/TPROF interpolation loop in lwrad_prep.

    The scalar linear search for klev (first standard level below pl,
    search starting at the SECOND level) becomes a searchsorted on the
    negated (ascending) pressure table with the identical strict-<
    comparison; the klev==NPROFLEVS branch reduces exactly to TPROF(60)
    (wght = 0, vark1 = vark), selected by mask.  Lanes taking that
    branch gather through a clamped safe index; their interpolated
    values are discarded (the denominators are table constants, never
    zero, so no out-of-domain arithmetic happens on any lane).

    Host arrays are re-entered with ``xp`` bound to CuPy when a device is
    present (S3b of the perf wave): the body below is already generic in
    ``xp`` and fully vectorised, so the device path executes the
    identical float32 expression, and searchsorted on the negated table
    performs the identical strict comparisons on identical bits.
    """
    if xp is np:
        device = _perfwave_device_xp()
        if device is not None:
            return device.asnumpy(_varint_b(device, device.asarray(plev)))
    f32 = np.float32
    tabs = _tables(xp)
    pprof = tabs["pprof"]
    tprof = tabs["tprof"]
    deep = pprof[_NPROFLEVS - 1] < plev
    j0 = xp.searchsorted(tabs["neg_pprof"], -plev, side="right")
    j0 = xp.minimum(xp.maximum(j0, 1), _NPROFLEVS - 1)
    vark = tprof[j0 - 1]
    vark1 = tprof[j0]
    wght = ((plev - pprof[j0 - 1]).astype(f32)
            / (pprof[j0] - pprof[j0 - 1]).astype(f32)).astype(f32)
    interp = ((wght * (vark1 - vark).astype(f32)).astype(f32)
              + vark).astype(f32)
    return xp.where(deep, interp, tprof[_NPROFLEVS - 1]).astype(f32)


def _mcica_column_groups(idcor, icld, lat_host):
    """Column index groups over which one generator call is per-column
    exact, or None when a single whole-batch call is (lat is read into
    the outputs only under idcor==1 with the exponential overlaps)."""
    if int(idcor) == 1 and int(icld) in (4, 5):
        pats = np.ascontiguousarray(lat_host).view(np.uint32)
        uniq, inv = np.unique(pats, return_inverse=True)
        if uniq.size > 1:
            return [np.nonzero(inv == i)[0] for i in range(uniq.size)]
    return None


def _mcica_host_batched(generate, ncol, nlay, icld, permuteseed, idcor,
                        juldat, lat_host, profs, bands, hgt):
    """One-or-few-call McICA over a batch, on the host.

    ``profs`` are the eight (ncol, nlay) per-column arrays in generator
    order (play, cldfrac, ciwp, clwp, cswp, rei, rel, res); ``bands``
    the (nbnd, ncol, nlay) band arrays (LW: tauc; SW: tauc, ssac, asmc,
    fsfc).  Splits by xlat bit pattern exactly when the scalar lat dummy
    can reach the outputs (see :func:`_mcica_column_groups`); per-column
    results are bitwise independent of the grouping (gated).
    """
    def _call(cols):
        if cols is None:
            return generate(
                1, ncol, nlay, icld, permuteseed, 0, *profs, *bands, hgt,
                int(idcor), juldat, F(lat_host[0]))
        return generate(
            1, cols.size, nlay, icld, permuteseed, 0,
            *(a[cols] for a in profs), *(a[:, cols, :] for a in bands),
            hgt[cols], int(idcor), juldat, F(lat_host[cols[0]]))

    groups = _mcica_column_groups(idcor, icld, lat_host)
    if groups is None:
        return _call(None)
    out = None
    for cols in groups:
        sub = _call(cols)
        if out is None:
            out = {k: np.empty((v.shape[0], ncol, nlay) if v.ndim == 3
                               else (ncol, nlay), np.float32)
                   for k, v in sub.items()}
        for k, v in sub.items():
            if v.ndim == 3:
                out[k][:, cols, :] = v
            else:
                out[k][cols] = v
    return out


def lwrad_prep_batch(*, p3d, p8w, t3d, t8w, dz8w,
                     qv3d, qc3d=None, qr3d=None, qi3d=None, qs3d=None,
                     qg3d=None, cldfra3d=None, o33d=None,
                     re_cloud=None, re_ice=None, re_snow=None,
                     tsk, emiss, xland, xice, snow, xlat,
                     icloud, warm_rain, cldovrlp, idcor, o3input,
                     has_reqc, has_reqi, has_reqs,
                     f_qc=True, f_qr=True, f_qi=True, f_qs=True, f_qg=True,
                     yr, julian, nlayers, mp_physics=0, g=9.81,
                     subcolumn_generator=None, trace_gas_overrides=None,
                     rrtmg_cloud_optics_form="wrf_461"):
    """Column-vectorized twin of :func:`lwrad_prep` (bitwise per column).

    Profiles are (ncol, kte) / (ncol, kte+1), surface fields (ncol,)
    (scalars broadcast); configuration args are python scalars shared by
    the batch (Section 11 flag-sharing contract).  Returns the scalar
    entry's dict with a leading ncol axis everywhere: McICA arrays
    (ngptlw, ncol, nlayers), reicmcl/relqmcl/resnmcl (ncol, nlayers),
    emis (ncol, nbndlw), tauaer (ncol, nlayers, nbndlw), tsfc (ncol,) --
    exactly the batched LW engine's input layout.  Accepts cupy arrays
    (the McICA stage still runs on the host).

    ``subcolumn_generator`` (default None -> the certified NumPy
    :func:`woof.core.rrtmg_mcica.generate_lw_subcolumns`) injects a
    signature-identical generator; the legacy forecast adapter passes
    the bit-gated device twin so the McICA slabs come back as device
    arrays and the NumPy generator never runs on the forecast path.
    """
    cloud_form_is_fork(rrtmg_cloud_optics_form)
    _require_radii("lwrad_prep_batch", has_reqc, has_reqi, has_reqs,
                   re_cloud, re_ice, re_snow)
    _mp_guard(mp_physics)
    xp = _get_xp(p3d, p8w, t3d, t8w, dz8w, qv3d)
    f32 = np.float32
    p3d = xp.asarray(p3d, f32)
    if p3d.ndim != 2:
        raise ValueError("lwrad_prep_batch expects (ncol, kte) profiles")
    ncol, kte = p3d.shape
    if ncol == 0:
        raise ValueError("empty batch (ncol == 0)")
    nlayers = int(nlayers)
    if nlayers < kte + 1:
        raise ValueError("nlayers must be at least kte+1")
    p8w = _prof_b(xp, p8w, ncol, kte + 1, "p8w")
    t3d = _prof_b(xp, t3d, ncol, kte, "t3d")
    t8w = _prof_b(xp, t8w, ncol, kte + 1, "t8w")
    dz8w = _prof_b(xp, dz8w, ncol, kte, "dz8w")
    qv3d = _prof_b(xp, qv3d, ncol, kte, "qv3d")
    zeros = xp.zeros((ncol, kte), f32)
    qc3d = zeros if qc3d is None else _prof_b(xp, qc3d, ncol, kte, "qc3d")
    qr3d = zeros if qr3d is None else _prof_b(xp, qr3d, ncol, kte, "qr3d")
    qi3d = zeros if qi3d is None else _prof_b(xp, qi3d, ncol, kte, "qi3d")
    qs3d = zeros if qs3d is None else _prof_b(xp, qs3d, ncol, kte, "qs3d")
    qg3d = zeros if qg3d is None else _prof_b(xp, qg3d, ncol, kte, "qg3d")
    cldfra3d = (zeros if cldfra3d is None
                else _prof_b(xp, cldfra3d, ncol, kte, "cldfra3d"))
    re_cloud = (zeros if re_cloud is None
                else _prof_b(xp, re_cloud, ncol, kte, "re_cloud"))
    re_ice = (zeros if re_ice is None
              else _prof_b(xp, re_ice, ncol, kte, "re_ice"))
    re_snow = (zeros if re_snow is None
               else _prof_b(xp, re_snow, ncol, kte, "re_snow"))
    tsk = _surf_b(xp, tsk, ncol, "tsk")
    emiss = _surf_b(xp, emiss, ncol, "emiss")
    xland = _surf_b(xp, xland, ncol, "xland")
    xice = _surf_b(xp, xice, ncol, "xice")
    snow = _surf_b(xp, snow, ncol, "snow")
    xlat = _surf_b(xp, xlat, ncol, "xlat")

    gases = lw_trace_gases(yr, trace_gas_overrides)

    pw1d = (p8w / F("100.0")).astype(f32)
    tw1d = t8w.copy()
    t1d = t3d.copy()
    p1d = (p3d / F("100.0")).astype(f32)
    dz1d = dz8w.copy()

    if o3input == 2:
        o31d = _prof_b(xp, o33d, ncol, kte, "o33d").copy()
    else:
        o31d = xp.zeros((ncol, kte), f32)

    # The moisture -> effective-radii -> cloud-path block ALWAYS runs on
    # the host: mixing ratios traverse the FP32 subnormal range on real
    # columns and every cupy-compiled device op flushes subnormals
    # except copies/where-selects (see the module docstring's cupy
    # bullet), so this is the one prep chain the device cannot
    # reproduce bitwise on this compile route.  For
    # numpy inputs _host_f32 is the identity and nothing changes.
    hb = _host_f32
    cldfra_h = hb(cldfra3d)
    xland_h, xice_h, snow_h = hb(xland), hb(xice), hb(snow)
    qv1d, qc1d, qr1d, qi1d, qs1d, qg1d, cldfra1d = _moist_prep_b(
        np, icloud, warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg, hb(t1d),
        hb(qv3d), hb(qc3d), hb(qr3d), hb(qi3d), hb(qs3d), hb(qg3d),
        cldfra_h, ferrier=_is_ferrier(mp_physics), eta_ferrier=(int(mp_physics)==95))

    icld = int(cldovrlp)
    juldat = int(F(julian))          # Fortran INT() on the REAL julian
    liqflglw = 1

    inflglw, iceflglw, recloud1d, reice1d, resnow1d, qs1d, qi1d = \
        _effective_radii_b(np, icloud, has_reqc, has_reqi, has_reqs,
                           hb(t3d), cldfra_h, xland_h, hb(re_cloud),
                           hb(re_ice), hb(re_snow), hb(qi3d), qs1d, qi1d)

    # ---- level/layer assembly with Cavallo buffer layers ----
    plev = xp.zeros((ncol, nlayers + 1), f32)
    tlev = xp.zeros((ncol, nlayers + 1), f32)
    play = xp.zeros((ncol, nlayers), f32)
    tlay = xp.zeros((ncol, nlayers), f32)
    hgt = xp.zeros((ncol, nlayers), f32)

    plev[:, 0] = pw1d[:, 0]
    tlev[:, 0] = tw1d[:, 0]
    tsfc = tsk.astype(f32)
    plev[:, 1:kte + 1] = pw1d[:, 1:kte + 1]
    tlev[:, 1:kte + 1] = tw1d[:, 1:kte + 1]
    play[:, :kte] = p1d
    tlay[:, :kte] = t1d
    pdel = (plev[:, :kte] - plev[:, 1:kte + 1]).astype(f32)

    h2ovmr = xp.zeros((ncol, nlayers), f32)
    h2ovmr[:, :kte] = xp.asarray((qv1d * _AMDW).astype(f32))
    co2vmr = xp.full((ncol, nlayers), gases["co2"], f32)
    o2vmr = xp.full((ncol, nlayers), gases["o2"], f32)
    ch4vmr = xp.full((ncol, nlayers), gases["ch4"], f32)
    n2ovmr = xp.full((ncol, nlayers), gases["n2o"], f32)
    cfc11vmr = xp.full((ncol, nlayers), gases["cfc11"], f32)
    cfc12vmr = xp.full((ncol, nlayers), gases["cfc12"], f32)
    cfc22vmr = xp.full((ncol, nlayers), gases["cfc22"], f32)
    ccl4vmr = xp.full((ncol, nlayers), gases["ccl4"], f32)
    h2ovmr[:, kte:] = h2ovmr[:, kte - 1:kte]

    dzsum = xp.zeros(ncol, f32)
    for k in range(kte):
        dz = dz1d[:, k]
        hgt[:, k] = (dzsum + (_HALF * dz).astype(f32)).astype(f32)
        dzsum = (dzsum + dz).astype(f32)

    # Buffer levels above the model top (Cavallo): 4-mb steps; dz stays
    # the last model layer's thickness, exactly like the scalar loop.
    # POSITIVITY: nlayers is sized from the nominal p_top but the
    # column tops in plev[:, kte] vary; a top shallower than n_extra-1
    # 4-mb steps would march plev/play nonpositive, log(play) in
    # setcoef goes NaN, laytrop miscounts, and the band kernels index
    # absa/selfref outside their tables (pool-layout-dependent reads).
    # The final interface is overwritten with zero below, so its
    # temporary subtraction does not constrain dp. WRF never meets
    # that state (its plev(kte+1) IS the grid-constant
    # p_top that sized nlayers); a genuinely shallow top retains the
    # existing conservative reduced-step fallback. In-contract
    # columns keep dp == DELTAP, bitwise.
    n_extra = nlayers - kte
    dp = DELTAP
    if n_extra > 0:
        shallow = plev[:, kte] <= F(DELTAP * F(n_extra - 1))
        dp = xp.where(
            shallow,
            xp.minimum(DELTAP, (plev[:, kte] / F(n_extra)).astype(f32)),
            DELTAP).astype(f32)
    for L in range(kte, nlayers):          # Fortran L = kte+1 .. nlayers
        plev[:, L + 1] = (plev[:, L] - dp).astype(f32)
        play[:, L] = (_HALF * (plev[:, L] + plev[:, L + 1]).astype(
            f32)).astype(f32)
        hgt[:, L] = (dzsum + (_HALF * dz).astype(f32)).astype(f32)
        dzsum = (dzsum + dz).astype(f32)
    plev[:, nlayers] = F("0.00")
    play[:, nlayers - 1] = (_HALF * (plev[:, nlayers - 1]
                                     + plev[:, nlayers]).astype(
        f32)).astype(f32)

    varint = _varint_b(xp, plev)

    # WRF oddity preserved: the overwrite starts at tlev(kte+1)/tlay(kte)
    # and anchors at the interface BELOW the model top.  All overwritten
    # tlev values depend only on varint + anchor, so the scalar L-loop
    # flattens to two slice assignments (tlev first, then tlay from the
    # updated tlev, whose L-2 term at the first row is the untouched
    # t8w interface value).
    anchor = (tlev[:, kte - 1] - varint[:, kte - 1]).astype(f32)
    tlev[:, kte:nlayers + 1] = (varint[:, kte:nlayers + 1]
                                + anchor[:, None]).astype(f32)
    tlay[:, kte - 1:nlayers] = (_HALF * (tlev[:, kte:nlayers + 1]
                                         + tlev[:, kte - 1:nlayers]
                                         ).astype(f32)).astype(f32)

    # Ozone profile including the buffer layers.
    o3mmr = _o3data_b(xp, plev)            # (ncol, nlayers)
    o3vmr = xp.zeros((ncol, nlayers), f32)
    if o3input == 2:
        o3vmr[:, :kte] = o31d
        shift_base = (o3mmr[:, kte - 1] * _AMDO).astype(f32)
        base = (o31d[:, kte - 1] - shift_base).astype(f32)
        clim = (o3mmr[:, kte:] * _AMDO).astype(f32)
        v = (base[:, None] + clim).astype(f32)
        o3vmr[:, kte:] = xp.where(v <= F("0.0"), clim, v).astype(f32)
    else:
        o3vmr = (o3mmr * _AMDO).astype(f32)
        o31d = o3vmr[:, :kte].copy()       # WRF writes this back to O33D

    emis = xp.empty((ncol, NBNDLW), f32)
    emis[...] = emiss[:, None]

    # ---- cloud properties on model layers (inflglw > 0 always here);
    # host block (see above), device copies for the returned arrays ----
    cldfrac_h = np.zeros((ncol, nlayers), f32)
    cldfrac_h[:, :kte] = cldfra1d
    clwpth_h = np.zeros((ncol, nlayers), f32)
    ciwpth_h = np.zeros((ncol, nlayers), f32)
    cswpth_h = np.zeros((ncol, nlayers), f32)
    rel_h = np.full((ncol, nlayers), F("10.0"), f32)
    rei_h = np.full((ncol, nlayers), F("10.0"), f32)
    res_h = np.full((ncol, nlayers), F("10.0"), f32)
    (clwpth_h[:, :kte], ciwpth_h[:, :kte], cswpth_h[:, :kte],
     rel_h[:, :kte], rei_h[:, :kte], res_h[:, :kte], _resnow) = \
        _cloud_properties_b(
            np, iceflglw, inflglw, qc1d, qi1d, qs1d, cldfra1d, hb(pdel),
            hb(tlay[:, :kte]), g, xland_h, xice_h, snow_h,
            recloud1d, reice1d, resnow1d,
            cloud_form=rrtmg_cloud_optics_form)
    cldfrac = xp.asarray(cldfrac_h)    # the only cloud array returned
    taucld_h = np.zeros((NBNDLW, ncol, nlayers), f32)

    # ---- McICA subcolumn generator (permuteseed = 150), on the host ----
    mc = _mcica_host_batched(
        (generate_lw_subcolumns if subcolumn_generator is None
         else subcolumn_generator),
        ncol, nlayers, icld, LW_PERMUTESEED,
        int(idcor), juldat, np.ascontiguousarray(hb(xlat)),
        (hb(play), cldfrac_h, ciwpth_h, clwpth_h, cswpth_h, rei_h,
         rel_h, res_h),
        (taucld_h,), hb(hgt))
    if xp is not np:
        mc = {k: xp.asarray(v) for k, v in mc.items()}

    tauaer = xp.zeros((ncol, nlayers, NBNDLW), f32)

    return {
        "ncol": ncol, "nlay": nlayers, "icld": icld,
        "play": play, "plev": plev, "tlay": tlay, "tlev": tlev,
        "tsfc": tsfc,
        "h2ovmr": h2ovmr, "o3vmr": o3vmr, "co2vmr": co2vmr,
        "ch4vmr": ch4vmr, "n2ovmr": n2ovmr, "o2vmr": o2vmr,
        "cfc11vmr": cfc11vmr, "cfc12vmr": cfc12vmr, "cfc22vmr": cfc22vmr,
        "ccl4vmr": ccl4vmr, "emis": emis,
        "inflglw": inflglw, "iceflglw": iceflglw, "liqflglw": liqflglw,
        "cldfmcl": mc["cldfmcl"], "taucmcl": mc["taucmcl"],
        "ciwpmcl": mc["ciwpmcl"], "clwpmcl": mc["clwpmcl"],
        "cswpmcl": mc["cswpmcl"],
        "reicmcl": mc["reicmcl"], "relqmcl": mc["relqmcl"],
        "resnmcl": mc["resnmcl"],
        "tauaer": tauaer,
        # extras (not rrtmg_lw dummies)
        "o31d": o31d, "hgt": hgt, "cldfrac": cldfrac, "juldat": juldat,
        "pdel": pdel,
    }


def lwrad_outputs_batch(*, uflx, dflx, hr, uflxc, dflxc, hrc, pi3d,
                        uflxcln=None, dflxcln=None, calc_clean_atm_diag=0):
    """Column-vectorized twin of :func:`lwrad_outputs` (bitwise per
    column).  Fluxes are (ncol, nlayers+1), heating rates (ncol,
    nlayers), pi3d (ncol, kte); every output gains a leading ncol axis.
    """
    xp = _get_xp(uflx, dflx, hr, uflxc, dflxc, hrc, pi3d)
    f32 = np.float32
    uflx = xp.asarray(uflx, f32)
    dflx = xp.asarray(dflx, f32)
    uflxc = xp.asarray(uflxc, f32)
    dflxc = xp.asarray(dflxc, f32)
    hr = xp.asarray(hr, f32)
    hrc = xp.asarray(hrc, f32)
    pi3d = xp.asarray(pi3d, f32)
    if pi3d.ndim != 2:
        raise ValueError("lwrad_outputs_batch expects (ncol, kte) pi3d")
    kte = pi3d.shape[1]
    ntop = uflx.shape[1] - 1               # nlayers+1-th Fortran level

    out = {
        "glw": dflx[:, 0].copy(),
        "olr": uflx[:, ntop].copy(),
        "lwcf": (uflxc[:, ntop] - uflx[:, ntop]).astype(f32),
        "lwupt": uflx[:, ntop].copy(), "lwuptc": uflxc[:, ntop].copy(),
        "lwdnt": dflx[:, ntop].copy(), "lwdntc": dflxc[:, ntop].copy(),
        "lwupb": uflx[:, 0].copy(), "lwupbc": uflxc[:, 0].copy(),
        "lwdnb": dflx[:, 0].copy(), "lwdnbc": dflxc[:, 0].copy(),
        "lwupflx": uflx[:, :kte + 2].copy(),
        "lwupflxc": uflxc[:, :kte + 2].copy(),
        "lwdnflx": dflx[:, :kte + 2].copy(),
        "lwdnflxc": dflxc[:, :kte + 2].copy(),
    }
    if calc_clean_atm_diag > 0 and uflxcln is not None:
        uflxcln = xp.asarray(uflxcln, f32)
        dflxcln = xp.asarray(dflxcln, f32)
        out["lwuptcln"] = uflxcln[:, ntop].copy()
        out["lwdntcln"] = dflxcln[:, ntop].copy()
        out["lwupbcln"] = uflxcln[:, 0].copy()
        out["lwdnbcln"] = dflxcln[:, 0].copy()

    tten = (hr[:, :kte] / F("86400.0")).astype(f32)
    out["rthratenlw"] = (tten / pi3d).astype(f32)
    tten = (hrc[:, :kte] / F("86400.0")).astype(f32)
    out["rthratenlwc"] = (tten / pi3d).astype(f32)
    return out


def swrad_prep_batch(*, p3d, p8w, t3d, t8w, dz8w,
                     qv3d, qc3d=None, qr3d=None, qi3d=None, qs3d=None,
                     qg3d=None, cldfra3d=None, o33d=None,
                     re_cloud=None, re_ice=None, re_snow=None,
                     tsk, albedo, xland, xice, snow, xlat,
                     xcoszen, solcon, obscur=0.0,
                     icloud, warm_rain, cldovrlp, idcor, o3input,
                     has_reqc, has_reqi, has_reqs,
                     f_qc=True, f_qr=True, f_qi=True, f_qs=True, f_qg=True,
                     yr, julian, mp_physics=0, g=9.81,
                     sf_surface_physics=2, spectral_albedo=None, subcolumn_generator=None, trace_gas_overrides=None,
                     rrtmg_cloud_optics_form="wrf_461"):
    """Column-vectorized twin of :func:`swrad_prep` (bitwise per column).

    Day contract: takes PRE-GATHERED day columns and raises ValueError
    if any xcoszen <= 0 -- the caller gathers day columns (e.g. cols =
    nonzero(xcoszen > 0)) and routes the complement through
    :func:`swrad_night_outputs` / :func:`swrad_night_outputs_batch`,
    exactly like WRF's per-column dorrsw gate.  Shapes/flag sharing as
    :func:`lwrad_prep_batch`; McICA arrays come back (ngptsw, ncol,
    nlay), mcica_inputs carries the batched call record with lat as the
    (ncol,) latitude vector.  ``subcolumn_generator`` injects a
    signature-identical McICA generator exactly as in
    :func:`lwrad_prep_batch` (default: the NumPy
    :func:`woof.core.rrtmg_mcica.generate_sw_subcolumns`).
    """
    fork_cloud = cloud_form_is_fork(rrtmg_cloud_optics_form)
    _require_radii("swrad_prep_batch", has_reqc, has_reqi, has_reqs,
                   re_cloud, re_ice, re_snow)
    _mp_guard(mp_physics)
    _require_spectral_albedo(sf_surface_physics, spectral_albedo)

    xp = _get_xp(p3d, p8w, t3d, t8w, dz8w, qv3d)
    f32 = np.float32
    p3d = xp.asarray(p3d, f32)
    if p3d.ndim != 2:
        raise ValueError("swrad_prep_batch expects (ncol, kte) profiles")
    ncol, kte = p3d.shape
    if ncol == 0:
        raise ValueError("empty batch (ncol == 0)")

    xcoszen = _surf_b(xp, xcoszen, ncol, "xcoszen")
    coszr = xcoszen.astype(f32)
    coszrs = xcoszen.astype(f32)
    if bool((coszrs <= F("0.0")).any()):
        raise ValueError(
            "night column (coszen <= 0) in batch: swrad_prep_batch takes "
            "pre-gathered day columns only; route night columns through "
            "swrad_night_outputs / swrad_night_outputs_batch")

    p8w = _prof_b(xp, p8w, ncol, kte + 1, "p8w")
    t3d = _prof_b(xp, t3d, ncol, kte, "t3d")
    t8w = _prof_b(xp, t8w, ncol, kte + 1, "t8w")
    dz8w = _prof_b(xp, dz8w, ncol, kte, "dz8w")
    qv3d = _prof_b(xp, qv3d, ncol, kte, "qv3d")
    zeros = xp.zeros((ncol, kte), f32)
    qc3d = zeros if qc3d is None else _prof_b(xp, qc3d, ncol, kte, "qc3d")
    qr3d = zeros if qr3d is None else _prof_b(xp, qr3d, ncol, kte, "qr3d")
    qi3d = zeros if qi3d is None else _prof_b(xp, qi3d, ncol, kte, "qi3d")
    qs3d = zeros if qs3d is None else _prof_b(xp, qs3d, ncol, kte, "qs3d")
    qg3d = zeros if qg3d is None else _prof_b(xp, qg3d, ncol, kte, "qg3d")
    cldfra3d = (zeros if cldfra3d is None
                else _prof_b(xp, cldfra3d, ncol, kte, "cldfra3d"))
    re_cloud = (zeros if re_cloud is None
                else _prof_b(xp, re_cloud, ncol, kte, "re_cloud"))
    re_ice = (zeros if re_ice is None
              else _prof_b(xp, re_ice, ncol, kte, "re_ice"))
    re_snow = (zeros if re_snow is None
               else _prof_b(xp, re_snow, ncol, kte, "re_snow"))
    tsk = _surf_b(xp, tsk, ncol, "tsk")
    albedo = _surf_b(xp, albedo, ncol, "albedo")
    xland = _surf_b(xp, xland, ncol, "xland")
    xice = _surf_b(xp, xice, ncol, "xice")
    snow = _surf_b(xp, snow, ncol, "snow")
    xlat = _surf_b(xp, xlat, ncol, "xlat")
    solcon = _surf_b(xp, solcon, ncol, "solcon")
    obscur = _surf_b(xp, obscur, ncol, "obscur")

    co2, ch4, n2o, o2 = option4_trace_gases(yr, trace_gas_overrides)

    pw1d = (p8w / F("100.0")).astype(f32)
    tw1d = t8w.copy()
    t1d = t3d.copy()
    p1d = (p3d / F("100.0")).astype(f32)
    dz1d = dz8w.copy()

    if o3input == 2:
        o31d = _prof_b(xp, o33d, ncol, kte, "o33d").copy()
    else:
        o31d = xp.zeros((ncol, kte), f32)

    # Host block for the subnormal-hazard chain, exactly as in
    # lwrad_prep_batch (see the comment there and the module docstring).
    hb = _host_f32
    cldfra_h = hb(cldfra3d)
    xland_h, xice_h, snow_h = hb(xland), hb(xice), hb(snow)
    qv1d, qc1d, qr1d, qi1d, qs1d, qg1d, cldfra1d = _moist_prep_b(
        np, icloud, warm_rain, f_qc, f_qr, f_qi, f_qs, f_qg, hb(t1d),
        hb(qv3d), hb(qc3d), hb(qr3d), hb(qi3d), hb(qs3d), hb(qg3d),
        cldfra_h, ferrier=_is_ferrier(mp_physics), eta_ferrier=(int(mp_physics)==95))

    nlay = kte + 1
    icld = int(cldovrlp)
    juldat = int(F(julian))
    liqflgsw = 1

    inflgsw, iceflgsw, recloud1d, reice1d, resnow1d, qs1d, qi1d = \
        _effective_radii_b(np, icloud, has_reqc, has_reqi, has_reqs,
                           hb(t3d), cldfra_h, xland_h, hb(re_cloud),
                           hb(re_ice), hb(re_snow), hb(qi3d), qs1d, qi1d, sw_cloud_fallback=fork_cloud)

    coszen = coszrs
    scon = (solcon * (_ONE - obscur).astype(f32)).astype(f32)
    dyofyr = 0                 # WRF: solcon already carries eccentricity
    adjes = _ONE

    # ---- level/layer assembly with ONE extra layer ----
    plev = xp.zeros((ncol, nlay + 1), f32)
    tlev = xp.zeros((ncol, nlay + 1), f32)
    play = xp.zeros((ncol, nlay), f32)
    tlay = xp.zeros((ncol, nlay), f32)
    hgt = xp.zeros((ncol, nlay), f32)

    plev[:, 0] = pw1d[:, 0]
    tlev[:, 0] = tw1d[:, 0]
    tsfc = tsk.astype(f32)
    plev[:, 1:kte + 1] = pw1d[:, 1:kte + 1]
    tlev[:, 1:kte + 1] = tw1d[:, 1:kte + 1]
    play[:, :kte] = p1d
    tlay[:, :kte] = t1d
    pdel = (plev[:, :kte] - plev[:, 1:kte + 1]).astype(f32)

    h2ovmr = xp.zeros((ncol, nlay), f32)
    h2ovmr[:, :kte] = xp.asarray((qv1d * _AMDW).astype(f32))
    co2vmr = xp.full((ncol, nlay), co2, f32)
    o2vmr = xp.full((ncol, nlay), o2, f32)
    ch4vmr = xp.full((ncol, nlay), ch4, f32)
    n2ovmr = xp.full((ncol, nlay), n2o, f32)

    dzsum = xp.zeros(ncol, f32)
    for k in range(kte):
        dz = dz1d[:, k]
        hgt[:, k] = (dzsum + (_HALF * dz).astype(f32)).astype(f32)
        dzsum = (dzsum + dz).astype(f32)

    play[:, kte] = (_HALF * plev[:, kte]).astype(f32)
    tlay[:, kte] = (tlev[:, kte] + _ZERO).astype(f32)
    plev[:, kte + 1] = F("1.0e-5")
    tlev[:, kte + 1] = (tlev[:, kte] + _ZERO).astype(f32)
    h2ovmr[:, kte] = h2ovmr[:, kte - 1]
    hgt[:, kte] = (dzsum + (_HALF * dz).astype(f32)).astype(f32)

    o3mmr = _o3data_b(xp, plev)            # (ncol, kte+1) layer values
    o3vmr = xp.zeros((ncol, nlay), f32)
    if o3input == 2:
        o3vmr[:, :kte] = o31d
        clim = (o3mmr[:, kte] * _AMDO).astype(f32)
        v = ((o31d[:, kte - 1] - (o3mmr[:, kte - 1] * _AMDO).astype(
            f32)).astype(f32) + clim).astype(f32)
        o3vmr[:, kte] = xp.where(v <= F("0.0"), clim, v).astype(f32)
    else:
        o3vmr = (o3mmr * _AMDO).astype(f32)
        o31d = o3vmr[:, :kte].copy()

    # Albedo split (non-SSiB branch): all four components identical.
    asdir = (albedo.astype(f32) if spectral_albedo is None else
             _surf_b(xp, spectral_albedo["asdir"], ncol, "asdir").astype(f32))
    asdif = (albedo.astype(f32) if spectral_albedo is None else
             _surf_b(xp, spectral_albedo["asdif"], ncol, "asdif").astype(f32))
    aldir = (albedo.astype(f32) if spectral_albedo is None else
             _surf_b(xp, spectral_albedo["aldir"], ncol, "aldir").astype(f32))
    aldif = (albedo.astype(f32) if spectral_albedo is None else
             _surf_b(xp, spectral_albedo["aldif"], ncol, "aldif").astype(f32))

    # ---- cloud properties on model layers (inflgsw > 0 always here);
    # host block (see above), the mcica_inputs record keeps xp arrays ----
    cldfrac_h = np.zeros((ncol, nlay), f32)
    cldfrac_h[:, :kte] = cldfra1d
    clwpth_h = np.zeros((ncol, nlay), f32)
    ciwpth_h = np.zeros((ncol, nlay), f32)
    cswpth_h = np.zeros((ncol, nlay), f32)
    rel_h = np.full((ncol, nlay), F("10.0"), f32)
    rei_h = np.full((ncol, nlay), F("10.0"), f32)
    res_h = np.full((ncol, nlay), F("10.0"), f32)
    (clwpth_h[:, :kte], ciwpth_h[:, :kte], cswpth_h[:, :kte],
     rel_h[:, :kte], rei_h[:, :kte], res_h[:, :kte], _resnow) = \
        _cloud_properties_b(
            np, iceflgsw, inflgsw, qc1d, qi1d, qs1d, cldfra1d, hb(pdel),
            hb(tlay[:, :kte]), g, xland_h, xice_h, snow_h,
            recloud1d, reice1d, resnow1d,
            cloud_form=rrtmg_cloud_optics_form)
    taucld_h = np.zeros((NBNDSW, ncol, nlay), f32)
    ssacld_h = np.ones((NBNDSW, ncol, nlay), f32)
    asmcld_h = np.zeros((NBNDSW, ncol, nlay), f32)
    fsfcld_h = np.zeros((NBNDSW, ncol, nlay), f32)

    play_h = hb(play)
    hgt_h = hb(hgt)
    mcica_inputs = {
        "play": play.copy(), "cldfrac": xp.asarray(cldfrac_h),
        "ciwpth": xp.asarray(ciwpth_h), "clwpth": xp.asarray(clwpth_h),
        "cswpth": xp.asarray(cswpth_h), "rei": xp.asarray(rei_h),
        "rel": xp.asarray(rel_h), "res": xp.asarray(res_h),
        "hgt": hgt.copy(), "icld": icld,
        "idcor": int(idcor), "juldat": juldat, "lat": xlat.astype(f32),
        "permuteseed": SW_PERMUTESEED, "irng": 0,
    }

    mc = _mcica_host_batched(
        (generate_sw_subcolumns if subcolumn_generator is None
         else subcolumn_generator),
        ncol, nlay, icld, SW_PERMUTESEED,
        int(idcor), juldat, np.ascontiguousarray(hb(xlat)),
        (play_h, cldfrac_h, ciwpth_h, clwpth_h, cswpth_h, rei_h, rel_h,
         res_h),
        (taucld_h, ssacld_h, asmcld_h, fsfcld_h), hgt_h)
    if xp is not np:
        mc = {k: xp.asarray(v) for k, v in mc.items()}

    return {
        "ncol": ncol, "nlay": nlay, "icld": icld,
        "play": play, "plev": plev, "tlay": tlay, "tlev": tlev,
        "tsfc": tsfc,
        "h2ovmr": h2ovmr, "o3vmr": o3vmr, "co2vmr": co2vmr,
        "ch4vmr": ch4vmr, "n2ovmr": n2ovmr, "o2vmr": o2vmr,
        "asdir": asdir, "asdif": asdif, "aldir": aldir, "aldif": aldif,
        "coszen": coszen, "adjes": adjes, "dyofyr": dyofyr, "scon": scon,
        "inflgsw": inflgsw, "iceflgsw": iceflgsw, "liqflgsw": liqflgsw,
        "cldfmcl": mc["cldfmcl"], "taucmcl": mc["taucmcl"],
        "ssacmcl": mc["ssacmcl"], "asmcmcl": mc["asmcmcl"],
        "fsfcmcl": mc["fsfcmcl"], "ciwpmcl": mc["ciwpmcl"],
        "clwpmcl": mc["clwpmcl"], "cswpmcl": mc["cswpmcl"],
        "reicmcl": mc["reicmcl"], "relqmcl": mc["relqmcl"],
        "resnmcl": mc["resnmcl"],
        # WRF-level extras
        "coszr": coszr, "o31d": o31d, "juldat": juldat, "pdel": pdel,
        "mcica_inputs": mcica_inputs,
    }


def swrad_night_outputs_batch(xcoszen, calc_clean_atm_diag=0):
    """Column-vectorized twin of :func:`swrad_night_outputs`: COSZR is
    the (ncol,) xcoszen copy, every :data:`SW_NIGHT_ZEROED` diagnostic a
    (ncol,) zero vector; the :data:`SW_NIGHT_UNTOUCHED` fields stay
    deliberately absent."""
    xp = _get_xp(xcoszen)
    cz = xp.asarray(xcoszen, np.float32)
    if cz.ndim != 1:
        raise ValueError("xcoszen must be shape (ncol,)")
    ncol = cz.shape[0]
    out = {"coszr": cz.astype(np.float32)}
    for name in SW_NIGHT_ZEROED:
        out[name] = xp.zeros(ncol, np.float32)
    if calc_clean_atm_diag > 0:
        for name in ("swuptcln", "swdntcln", "swupbcln", "swdnbcln"):
            out[name] = xp.zeros(ncol, np.float32)
    return out
