"""EXPERIMENTAL (WOOF v1.2): radar observation operators H(x).

Two gridded operators, both pure functions of the model state, both
CuPy-first with a NumPy fallback, neither wired into any default route:

- :func:`simulated_reflectivity` -- 10 cm equivalent reflectivity (dBZ).
- :func:`radial_velocity` -- Doppler radial velocity (m/s) toward or away
  from a radar site, under 4/3-earth beam geometry.

They produce the model-grid half of the ``gpuwm-obs.radar-grid.v1`` obs
schema (gridded ``z_obs``/``vr_obs`` plus masks on the model grid), so an
ensemble filter can difference H(x) against the observation arrays
element-wise with no interpolation of its own.

Reflectivity: NO SECOND Z FORMULA
=================================

WOOF already computes reflectivity on the product side, and this module
does not reimplement it.  ``simulated_reflectivity`` is a thin,
shape-checking adapter over the existing authority:

- device path: ``woof.core.refl.compute_refl_10cm``
  (woof/core/refl.py:448) verbatim for schemes 1/6/8/10 -- the same call
  the microphysics adapters make for wrfout ``REFL_10CM`` -- and
  ``woof.core.nssl2_diagnostics.diagnose_radardd02_if_due`` for scheme 18,
  the same NSSL diagnostic the production coordinator runs;
- host path: the certified float64 column mirrors
  ``woof.verify.npref.np_refl10cm_morrison_column`` (npref.py:9204),
  ``np_refl10cm_wsm6_column`` (npref.py:9351) and
  ``np_refl10cm_kessler_column`` (npref.py:9460), applied column by
  column.  Thompson (8) and NSSL (18) have no host mirror and are CUDA
  only.

That authority is WRF v4.6.1 ``do_radar_ref=1`` / REFL_10CM, dispatched by
``mp_physics`` exactly as ``microphysics.apply`` dispatches, so the
operator inherits the ACTIVE scheme's own PSD assumptions rather than
imposing one scheme's assumptions on another:

- ``mp_physics=10`` -- Morrison ``refl10cm_hm``
  (phys/module_mp_morr_two_moment.F:4502-4675, wrapper floor :913-917).
  Two-moment rain/snow/graupel using the scheme's PROGNOSTIC number
  concentrations; ``morr_rimed_ice`` selects graupel vs hail density.
  Cloud water contributes nothing (the routine reads no cloud moment).
- ``mp_physics=6`` -- WSM6 ``refl10cm_wsm6``
  (phys/mp_wsm6.F90:2275-2444): fixed-intercept exponential PSDs,
  ``hail_opt`` selecting the rimed-ice density/intercept.
- ``mp_physics=8`` -- classic Thompson ``calc_refl10cm``
  (phys/module_mp_thompson.F:5710-6028), which additionally needs the
  scheme's same-call private graupel number moment.
- ``mp_physics=1`` -- Kessler has no reflectivity diagnostic in WRF, so
  the authority's documented fallback is the Smith et al. (1975)
  fixed-intercept (N0r = 8e6 m-4) rain-only Rayleigh form derived at
  woof/core/refl.py:30-43.
- ``mp_physics=18`` -- NSSL two-moment ``radardd02``
  (phys/module_mp_nssl_2mom.F): five ice categories, five predicted
  number moments and the graupel/hail volume moments, S-band, CUDA only.
  It is a genuine scheme port, not an adapter over one of the above, so
  it lives in ``woof.core.nssl2_diagnostics`` and is dispatched to
  verbatim here.  Two departures from the 1/6/8/10 family the caller must
  carry: radardd02 floors at **0 dBZ**, not -35, and it reads five ice
  categories plus their moments rather than the shared rain-first set.
- ``mp_physics=50`` -- P3, through
  ``woof.core.p3_device.reflectivity``: the Z half of the scheme's own
  final-diagnostics loop (WRF module_mp_p3.F:4722-4895), replayed on
  LOCAL copies in woof/core/kernels/p3_zdiag.cu over the forecast port's
  unchanged device helpers, so evaluating it moves nothing.  The forecast
  loop fuses that Z with a state update (sub-QSMALL dumps into vapour and
  theta, the rain-DSD and ice-number clamps), which is why this was a named
  refusal until the Z was lifted out.  Its one clear-air value is the
  scheme's zero-hydrometeor reflectivity, -36.9897 dBZ (the 1e-22 seeds at
  :2286-2287).  ``woof.core.refl.compute_refl_10cm`` still carries no
  mp=50 branch: its only P3-satisfiable arm is the rain-only Kessler
  fallback.

The 1/6/8/10/16/28 routes share the -35 dBZ floor, the 1e-9 kg/kg species
activity threshold, and the ``refl10cm_hm`` air-density diagnosis, so
their outputs are interchangeable in one obs array.  NSSL (18) is native
S-band with a 0 dBZ floor, and Milbrandt-Yau (9) is native with a -99 dBZ
floor; neither is floor-interchangeable with that family.

Documented simplifications
==========================

1. **Fall speed is not the per-scheme reflectivity-weighted integral.**
   The bound Z formulations compute Ze only; none of them exposes a
   reflectivity-weighted mean terminal velocity.  Rather than invent a
   fourth PSD integration, :func:`reflectivity_fall_speed` uses the
   standard radar-DA relation of Sun and Crook (1997, J. Atmos. Sci. 54,
   1642-1661; 1998, J. Atmos. Sci. 55, 835-852), the same closure
   WRF-DA's ``da_radial_velocity`` and Tong and Xue (2005, Mon. Wea.
   Rev. 133, 1789-1807) use, driven by the SAME dBZ field the bound Z
   operator produced:

       vt = 5.40 * (ps/p)**0.4 * 10**(0.125*(Z - 43.1)/17.5)  [m/s, down]

   ``ps`` is the SURFACE pressure, supplied by the caller, not the
   1000 hPa thermodynamic reference: that is what Sun and Crook wrote
   and what WRFDA's ``da_radial_velocity.inc`` implements, and the
   substitution is a systematic +4.3%/+9.3%/+15.3% error in ``vt`` over
   a 900/800/700 hPa surface.

   That form is Sun and Crook's ``vt = 5.40*a*(rho*qr)**0.125`` with
   ``rho*qr`` eliminated through their ``Z = 43.1 + 17.5*log10(rho*qr)``,
   and their Z leg is exactly the Marshall-Palmer/Smith-1975 PSD the
   Kessler fallback here implements -- so on ``mp_physics=1`` the fall
   speed and the reflectivity share one PSD (checked numerically in
   :func:`reflectivity_fall_speed`).  On the Morrison, WSM6 and Thompson
   paths the richer scheme PSD supplies Z while the RAIN fall speed is
   still read off the Marshall-Palmer rain calibration: it consumes that
   scheme's Z, but it is NOT a moment integral over that scheme's PSD.

   Frozen species no longer fall at the rain relation.  Given the state's
   species (``species=`` and ``inverse_density=``, which both callers in
   this tree pass), snow, graupel and hail each take the Lin, Farley and
   Orville (1983, J. Clim. Appl. Meteor. 22, 1065-1092) mass-weighted
   fall speed of an exponential size distribution, and the cell's fall
   speed is the mass-weighted blend of its active species, rain keeping
   Sun and Crook.  The rain relation alone, applied to dry snow aloft,
   made a 30 dBZ snow cell fall at about 5 m/s where snow falls at about
   1 m/s, and the radial velocity operator projected the difference onto
   every elevated beam.  Mass weighting is itself an approximation (a
   radar weights by reflectivity); it is stated, and it is far nearer
   than one rain relation for every species.

   Because a power law in Z does not vanish in clear air (it still gives
   ~1.7 m/s at the -35 dBZ floor), the fall speed is gated OFF wherever
   the reflectivity authority itself sees no active hydrometeor: the
   1e-9 kg/kg threshold applied PER SPECIES to qr, qs, qg and qh, active
   where ANY of them clears it, exactly as ``refl10cm_hm`` tests each
   category before adding its Ze.  Gating the SUM instead -- which this
   module did -- activates cells whose every species is individually
   inactive: four at 3e-10 kg/kg contribute no reflectivity at all and
   used to switch on a 5.244 m/s fall speed.  (WRFDA's Sun-Crook gates on
   rain alone above 1e-5 kg/kg; that is the other defensible authority
   and it is NOT this one, because the Z consumed here is the model
   scheme's multi-species Z and not a rain-only one.)  Callers who want a
   different closure pass ``fall_speed=<array>``; callers who want pure
   air motion pass ``fall_speed="none"``.

2. **Straight-ray 4/3-earth geometry, no beam-volume weighting.**  The
   operator evaluates the beam direction at each model mass point and
   projects the point wind onto it.  There is no Gaussian beam-power
   weighting across the sample volume and no along-beam averaging: at
   nest resolutions the model cell is smaller than the radar gate, which
   is a resolution mismatch the obs-error covariance must absorb, not
   something this operator claims to solve.

3. **No attenuation, no beam blockage, no ground clutter, no dual-pol.**

4. **Spherical earth.**  Radius ``1/gpuwm.core.constants.RERADIUS`` =
   6 370 000 m, the same value ``woof.static.projection.EARTH_RADIUS_M``
   uses, so the beam geometry is consistent with the projection the grid
   was built on.

EXPERIMENTAL.  Not covered by the v1 stability promise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from woof.core import constants as c

EXPERIMENTAL = True

#: Obs schema these operators are the model-side half of.
OBS_SCHEMA = "gpuwm-obs.radar-grid.v1"

#: Spherical earth radius (m).  ``1/RERADIUS`` is the same 6 370 000 m
#: that ``woof.static.projection.EARTH_RADIUS_M`` (module_llxy.F:152)
#: uses, spelled from the core constant so it is derived, never
#: hardcoded twice.
EARTH_RADIUS_M = 1.0 / c.RERADIUS

#: Standard-atmosphere effective-earth radius ratio for radar beam
#: refraction (Doviak and Zrnic, *Doppler Radar and Weather
#: Observations*, 2nd ed., eq. 2.28b).
EFFECTIVE_EARTH_RATIO = 4.0 / 3.0

#: Species activity threshold shared with the reflectivity authority
#: (``refl10cm_hm`` and every woof mirror gate on ``q > 1e-9``).  Applied
#: PER SPECIES, as the authority applies it -- see
#: :func:`precipitating_activity_mask`.
Q_ACTIVE_THRESHOLD = 1.0e-9

#: The dBZ value :func:`simulated_reflectivity` returns in fully clear air,
#: per scheme.  This is the operator's own floor, read off the routes
#: documented at the top of this module and pinned by
#: ``tests/test_da_obsop_thompson_gpu.py`` (mp8 = -35) and
#: ``tests/test_da_obsop_nssl_gpu.py`` (mp18 = 0).
#:
#: It exists as a lookup because clear-air ("zero") assimilation needs the
#: number as *data*: a clear-air observation is differenced against H(x),
#: so the observation must carry the same floor the operator produces or
#: two agreeing clear skies yield an innovation of 35 dB (-35 against
#: NSSL's 0), 64 dB (-35 against Milbrandt-Yau's -99) or 99 dB (0 against
#: -99).  Typing ``-35.0`` at a call site is the defect this table
#: prevents -- it is right for six of the schemes in it and silently
#: catastrophic for the other two.
#:
#: A scheme is in this table only if ONE number is the whole answer for it.
#: The schemes for which it is not are refused by name in
#: :data:`CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER`, which is a different
#: statement from being absent.
CLEAR_AIR_FLOOR_DBZ: dict[int, float] = {
    2: -35.0,
    # WSM7's own refl10cm kernel floors every clear cell at -35 dBZ.
    # No standalone DA route is added here; its native field is available
    # through the output-due WSM-family runner.
    24: -35.0,
    7: -35.0,
    1: -35.0,
    # WSM3 runs refl10cm_wsm6's kernel on its phase-split qr
    # (woof.core.refl.wsm3_precipitation_phases), which floors every cell
    # at -35 dBZ exactly as mp=6 does.
    3: -35.0,
    4: -35.0,  # wsm5_refl10cm_column final clamp.
    6: -35.0,
    # WSM6R's H(x) is the same refl10cm_wsm6 operator (woof.core.refl).
    106: -35.0,
    8: -35.0,
    # WDM6 shares the refl10cm family's floor and always did: its operator
    # arm is woof.core.refl.launch_refl10cm_wdm6, and
    # woof/core/kernels/wdm6_refl.cu:357 writes
    # ``fmaxf(-35.0f, 10*log10(zsum*1e18))`` in EVERY cell of every column,
    # so a hydrometeor-free cell reads exactly -35.0.  16 was the only
    # selector compute_refl_10cm dispatches that this table lacked; the
    # absence was drift, not a decision (audit R-015).
    16: -35.0,
    # SBU-YLin's Z is woof.core.refl's refl10cm_wsm6 operator over the
    # scheme's snow/graupel partition, so it takes that kernel's floor.
    13: -35.0,
    # Stock refl10cm_wdm5 and refl10cm_wdm7 clamp every cell to -35 dBZ.
    # CUDA: wdm_family_refl.cuh:560 and wdm7_refl.cu:901; the committed
    # strict native replays compare these diagnostic words as well.
    14: -35.0,
    26: -35.0,
    10: -35.0,
    40: -35.0,
    18: 0.0,
    38: -35.0,
    28: -35.0,
    # Goddard GCE: refl10cm_gsfc stores MAX(-35., dBZ) in every cell
    # (module_mp_gsfcgce.F:2899; kernels/gsfcgce_refl.cu), one number.
    97: -35.0,
    # Milbrandt-Yau reports ONE clear-air number and it is not -35.  Its Z
    # block (module_mp_milbrandt2mom.F:3400-3466, transcribed at
    # woof/core/kernels/milbrandt2_zet.cu) sums five species terms, takes
    # the zero-sum branch to ``MY2_minZET`` and stores
    # ``fmaxf(zet, MY2_minZET)`` with ``MY2_minZET = -99.0``
    # (kernels/milbrandt2.cu:308).  The launch covers every cell on every
    # call, so there is no hydrometeor-free skip and no initialisation
    # value left standing -- which is exactly what separates mp=9 from
    # the P3 forecast field, which reports two clear values (see mp=50),
    # two different clear-air values.  A -99 dBZ floor is NOT
    # interchangeable with the -35 family in one obs array: give
    # reflectivity observations a matching floor or mask the clear-air
    # cells, exactly as for NSSL's 0 dBZ.
    9: -99.0,
    # P3's operator (woof.core.p3_device.reflectivity, kernel
    # woof/core/kernels/p3_zdiag.cu) reads the scheme's two accumulator
    # seeds in every cell with no rain and no ice above QSMALL:
    # ze_rain = ze_ice = 1e-22 mm6 m-3 (module_mp_p3.F:2286-2287), so
    # 10*log10(2e-22*1e18) = -36.9897 dBZ, this float32 value exactly
    # (woof.core.p3_device.CLEAR_AIR_DBZ; tests/test_p3_reflectivity_gpu.py
    # holds the kernel to it).  The forecast REFL_10CM additionally holds
    # the -99.0 entry initialisation (:2278) through columns the step's
    # no-hydrometeor skip (:3972) never diagnoses; that sentinel is not a
    # reflectivity, and the OPERATOR does not reproduce it, which is what
    # gives P3 one floor.  Not interchangeable with the -35 family in one
    # obs array, as for NSSL's 0 and Milbrandt-Yau's -99.  mp=51 is the
    # same Z (rain and ice only; the predicted droplet number does not enter
    # it, :4722-4895), so the same operator and the same floor.
    50: -36.98970031738281,
    53: -36.98970031738281,
    # Ferrier-Aligo: EGCP01COLUMN initialises every level's DBZ_col to
    # DBZmin = -20 (module_mp_fer_hires.F:1191, :1279) and replaces it only
    # where Zrain+Zsnow exceeds Zmin = 0.01 mm6 m-3 (:2496), so clear air
    # reads -20 exactly (kernels/fer_hires.cu writes the same word).
    5: -20.0,
    51: -36.98970031738281,
    52: -36.98970031738281,
    # NTU (mp=56): WOOF's own 10 cm operator over the scheme's predicted
    # distributions (woof/core/kernels/ntu.cu ntu_refl_cell) floors at
    # the -35 dBZ family value in every cell.
    56: -35.0,
}
CLEAR_AIR_FLOOR_DBZ[15] = -20.0

#: Schemes whose H(x) has NO single clear-air value, and the paragraph that
#: says why for each.
#:
#: Absence from :data:`CLEAR_AIR_FLOOR_DBZ` says "nobody has read this
#: scheme's floor yet", which is an omission somebody can close by reading
#: the scheme.  These schemes are a different thing and must not be
#: reported as that one: no number belongs in the table for them, because
#: the operator does not produce one number.  Their refusal names the
#: mechanism, so a reader finds a decision instead of a gap and an editor
#: who reaches for ``-35.0`` finds out what it would cost.
CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER: dict[int, str] = {
    # Empty, and kept: the class is a real one (an operator that reports
    # more than one clear-air value) and its refusal below still speaks for
    # a scheme added to it.  P3 sat here while its only reflectivity was
    # the forecast field, which holds -36.9897 at a clear level and the
    # -99.0 entry sentinel through a never-diagnosed column; its operator
    # (CLEAR_AIR_FLOOR_DBZ[50]) reports one value.
}

def clear_air_floor_dbz(mp_physics: int) -> float:
    """The active scheme's clear-air H(x) value, or a refusal.

    Never falls back to the -35 dBZ majority: an unknown scheme is one
    whose floor nobody has read, and guessing it wrong is invisible in
    every diagnostic except the analysis increment.

    A scheme in :data:`CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER` is refused with
    ITS OWN reason rather than the generic one, because "this scheme's
    floor has not been read" and "this scheme has no single floor to read"
    ask the reader to do different things.

    BOTH refusals name the same two ways out (state the value, or turn the
    clear-air arm off).  The unrecorded-floor branch named neither, so a
    caller reached a dead end at the one door that raises this at
    configuration time -- ``RadarAssimilationConfig``, which re-raises this
    message as its own error and asks the H(x) route question first, so
    "state the value" cannot land on a scheme that has no operator at all.
    """

    key = int(mp_physics)
    reason = CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER.get(key)
    if reason is not None:
        raise ValueError(
            f"mp_physics={key} has no single clear-air reflectivity floor, "
            f"so none is recorded and none can be: {reason}. Until that "
            "exists, run this scheme's cycles with clear_air=False; an "
            "explicit clear_air_value_dbz is honoured but is a claim about "
            "H(x) that this operator can show to be false for part of "
            "every field")
    if key not in CLEAR_AIR_FLOOR_DBZ:
        raise ValueError(
            f"no clear-air reflectivity floor is recorded for mp_physics="
            f"{key}; known schemes are "
            f"{sorted(CLEAR_AIR_FLOOR_DBZ)}. The floor is not guessable -- "
            "it is -35 dBZ for the refl10cm family, 0 dBZ for NSSL and "
            "-99 dBZ for Milbrandt-Yau, and assimilating clear air against "
            "the wrong one manufactures a 35, 64 or 99 dB innovation -- "
            "whichever pair of those three was crossed -- wherever the sky "
            "is genuinely clear. Two ways out, the same two the "
            "no-single-floor refusal offers: state clear_air_value_dbz, "
            "which is honoured verbatim and recorded as this run's stated "
            "floor, or run this scheme's cycles with clear_air=False, which "
            "leaves radial velocity and echo exactly as configured")
    return CLEAR_AIR_FLOOR_DBZ[key]

#: Sun and Crook fall speed ``vt = 5.40*a*(rho*qr)**0.125`` with
#: ``a = (ps/p)**0.4`` on the SURFACE pressure and ``rho*qr`` in g m-3,
#: and the companion reflectivity relation
#: ``Z[dBZ] = 43.1 + 17.5*log10(rho*qr)`` used to eliminate ``rho*qr``.
#: See :func:`reflectivity_fall_speed` for the elimination and for the
#: check that the Z leg is the same Marshall-Palmer PSD the Kessler
#: fallback in woof/core/refl.py:30-43 implements.
SUN_CROOK_VT_COEFF = 5.40
SUN_CROOK_VT_EXPONENT = 0.125
SUN_CROOK_Z_INTERCEPT_DBZ = 43.1
SUN_CROOK_Z_SLOPE_DB = 17.5
FALL_SPEED_DENSITY_EXPONENT = 0.4

#: Lin, Farley and Orville (1983) exponential-PSD fall-speed constants
#: for the frozen species, the set ARPS and the Tong and Xue (2005) radar
#: EnKF use: ``(a [m^(1-b) s^-1], b, intercept N0 [m^-4], bulk density
#: [kg m^-3])`` for a power-law fall speed ``V = a D^b`` scaled by
#: ``(RHO0/rho)**0.5``.  Graupel is Rutledge and Hobbs (1984); hail is
#: Lin's ``V = (4 g rho_h D / (3 C_D rho))**0.5`` with ``C_D = 0.6``,
#: written in the same ``a D^b`` form at ``rho = RHO0``.
FROZEN_FALL_SPEED_CONSTANTS = {
    "qs": (11.72, 0.41, 3.0e6, 100.0),
    "qg": (19.3, 0.37, 4.0e6, 400.0),
    "qh": ((4.0 * 9.81 * 917.0 / (3.0 * 0.6 * 1.225)) ** 0.5, 0.5,
           4.0e4, 917.0),
}

#: Reference air density of the ``(RHO0/rho)**0.5`` fall-speed correction.
FALL_SPEED_RHO0 = 1.225

#: Default clamp on the diagnosed fall speed (m/s).  Large enough for
#: hail, small enough that a pathological Z cannot dominate the beam
#: projection.
FALL_SPEED_MAX_MS = 30.0

__all__ = [
    "EXPERIMENTAL", "OBS_SCHEMA", "EARTH_RADIUS_M", "EFFECTIVE_EARTH_RATIO",
    "RadarSite", "GridGeometry", "BeamGeometry",
    "simulated_reflectivity", "radial_velocity",
    "beam_geometry", "mass_point_heights", "mass_point_winds",
    "reflectivity_fall_speed", "precipitating_mixing_ratio",
    "precipitating_activity_mask", "PRECIPITATING_SPECIES",
    "Q_ACTIVE_THRESHOLD",
    "destagger_u", "destagger_v", "destagger_w", "earth_relative_winds",
]


# --------------------------------------------------------------------------
# array-module dispatch: CuPy first, NumPy fallback
# --------------------------------------------------------------------------

def _array_module(value):
    """CuPy for device arrays, NumPy for host arrays.

    Never imports CuPy to answer a question about a NumPy array, so a
    host-only caller (and a CPU-only test module) touches no device.
    """
    if isinstance(value, np.ndarray):
        return np
    cp = sys.modules.get("cupy")
    if cp is None:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception:
            return np
    try:
        if isinstance(value, cp.ndarray):
            return cp
    except Exception:
        return np
    return np


def _asarray64(xp, value):
    """Host-side float64 view for geometry arithmetic."""
    return np.asarray(_to_host(xp, value), dtype=np.float64)


def _to_host(xp, value):
    if xp is np:
        return value
    return xp.asnumpy(value)


# --------------------------------------------------------------------------
# site / grid geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RadarSite:
    """A radar's position.  Longitude east-positive, altitude m AMSL."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    name: str = ""

    def __post_init__(self) -> None:
        if not -90.0 <= float(self.latitude_deg) <= 90.0:
            raise ValueError(
                f"radar latitude out of range: {self.latitude_deg}")
        if not -360.0 <= float(self.longitude_deg) <= 360.0:
            raise ValueError(
                f"radar longitude out of range: {self.longitude_deg}")
        if not np.isfinite(float(self.altitude_m)):
            raise ValueError("radar altitude must be finite")


@dataclass(frozen=True)
class GridGeometry:
    """Where the model's mass points are, in earth coordinates.

    ``latitude_deg`` / ``longitude_deg`` are ``(ny, nx)``; ``height_m`` is
    ``(nz, ny, nx)`` geometric height AMSL at mass points.

    The model state does not carry its own geolocation -- ``DomainState``
    has no XLAT/XLONG, and every runtime consumer that needs latitude
    (the radiation solvers) takes it as an explicit argument
    (``woof/core/rrtmgp.py:1581`` ``latitude_deg``).  This dataclass
    follows that precedent rather than inventing state fields.
    """

    latitude_deg: object
    longitude_deg: object
    height_m: object

    def __post_init__(self) -> None:
        lat = np.asarray(_to_host(_array_module(self.latitude_deg),
                                  self.latitude_deg))
        lon = np.asarray(_to_host(_array_module(self.longitude_deg),
                                  self.longitude_deg))
        hgt = np.asarray(_to_host(_array_module(self.height_m),
                                  self.height_m))
        if lat.ndim != 2:
            raise ValueError(f"latitude_deg must be (ny, nx), got {lat.shape}")
        if lat.shape != lon.shape:
            raise ValueError(
                f"latitude_deg {lat.shape} and longitude_deg {lon.shape} "
                "must have the same shape")
        if hgt.ndim != 3:
            raise ValueError(f"height_m must be (nz, ny, nx), got {hgt.shape}")
        if hgt.shape[1:] != lat.shape:
            raise ValueError(
                f"height_m {hgt.shape} does not sit on the "
                f"{lat.shape} latitude/longitude grid")
        # Catch a transposed or swapped lat/lon pair here rather than
        # letting it produce a plausible-looking but wrong beam.
        if not np.all(np.isfinite(lat)) or np.any(np.abs(lat) > 90.0):
            raise ValueError(
                "latitude_deg must be finite and within +/-90 degrees "
                f"(got range {float(np.nanmin(lat))} to "
                f"{float(np.nanmax(lat))}); a swapped latitude/longitude "
                "pair is the usual cause")
        if not np.all(np.isfinite(lon)) or np.any(np.abs(lon) > 360.0):
            raise ValueError(
                "longitude_deg must be finite and within +/-360 degrees "
                f"(got range {float(np.nanmin(lon))} to "
                f"{float(np.nanmax(lon))})")
        if not np.all(np.isfinite(hgt)):
            raise ValueError("height_m must be finite everywhere")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(np.asarray(
            _to_host(_array_module(self.height_m), self.height_m)).shape)

    @classmethod
    def from_state(cls, state, latitude_deg=None, longitude_deg=None):
        """Build from a state plus explicit geolocation.

        Heights come from the state's geopotential; latitude/longitude
        must be supplied, or carried by the state as ``xlat``/``xlong``
        (some ingest paths attach them).  Fails closed otherwise --
        guessing a domain's position would silently produce a plausible,
        wrong beam.
        """
        if latitude_deg is None:
            latitude_deg = getattr(state, "xlat", None)
        if longitude_deg is None:
            longitude_deg = getattr(state, "xlong", None)
        if latitude_deg is None or longitude_deg is None:
            raise ValueError(
                "GridGeometry.from_state needs the domain's mass-point "
                "latitude and longitude: DomainState does not carry XLAT/"
                "XLONG, so pass latitude_deg=/longitude_deg= (for example "
                "from woof.static.projection ProjectedGrid.latlon_mass())")
        return cls(latitude_deg=latitude_deg, longitude_deg=longitude_deg,
                   height_m=mass_point_heights(state))

    @classmethod
    def from_target_grid(cls, grid):
        """Build from the radar lane's :class:`woof.obs.TargetGrid`.

        The observation side and the operator side must agree about where
        a model cell *is*, or an innovation is a difference between two
        different places.  ``TargetGrid`` carries layer-interface heights
        ``z_w`` ``(nz+1, ny, nx)`` and places every gate in the layer whose
        interfaces bracket it, in that gate's own column; the operator
        evaluates H(x) at mass points.  The mass point of that layer is
        the midpoint of the two interfaces that defined it, so this is the
        one reduction that keeps the two stages talking about one cell.

        Passing the same ``TargetGrid`` that wrote a
        ``gpuwm-obs.radar-grid.v1`` file is therefore the whole seam: read
        the file back with ``expected_grid_identity=grid.identity_sha256()``
        and both sides are pinned to the same columns by a digest.
        """

        z_w = np.asarray(grid.z_w, dtype=np.float64)
        return cls(latitude_deg=np.asarray(grid.lat, dtype=np.float64),
                   longitude_deg=np.asarray(grid.lon, dtype=np.float64),
                   height_m=0.5 * (z_w[:-1] + z_w[1:]))

    @classmethod
    def from_wrfout(cls, path, *, frame: int = 0):
        """Build from a run artifact -- the geometry supply for a cycle.

        Goes through :meth:`woof.obs.TargetGrid.from_wrfout`, which
        rebuilds the projection from the file's global attributes and then
        checks it against the file's own XLAT/XLONG, refusing a
        disagreement above 1e-5 deg.  Reading XLAT/XLONG straight out of
        the file would be shorter and would skip that check; the check is
        the reason a nest with an off-centre reference point does not
        quietly shift every observation by a cell.

        Imported lazily so ``woof.da.obsop`` keeps working with no
        netCDF4 present -- H(x) itself needs no file.
        """

        from woof.obs.target_grid import TargetGrid   # noqa: PLC0415

        return cls.from_target_grid(TargetGrid.from_wrfout(path,
                                                           frame=frame))


@dataclass(frozen=True)
class BeamGeometry:
    """Beam direction at every model mass point, in local ENU.

    ``sin_elevation`` / ``cos_elevation`` and ``sin_azimuth`` /
    ``cos_azimuth`` describe the unit vector pointing FROM the radar
    THROUGH the mass point, resolved in that point's own east/north/up
    frame.  Azimuth is measured clockwise from north, so the unit vector
    is ``(cos_el*sin_az, cos_el*cos_az, sin_el)``.

    All arrays are float64 ``(nz, ny, nx)`` except the azimuth pair,
    which is ``(ny, nx)`` -- azimuth does not vary with height along a
    vertical column.  ``slant_range_m`` is the straight-line distance in
    the effective-earth frame.
    """

    sin_elevation: np.ndarray
    cos_elevation: np.ndarray
    sin_azimuth: np.ndarray
    cos_azimuth: np.ndarray
    slant_range_m: np.ndarray
    ground_range_m: np.ndarray
    effective_earth_radius_m: float

    def unit_vector_enu(self):
        """(east, north, up) components, each broadcast to (nz, ny, nx)."""
        east = self.cos_elevation * self.sin_azimuth
        north = self.cos_elevation * self.cos_azimuth
        return east, north, self.sin_elevation


def mass_point_heights(state):
    """Geometric height AMSL ``(nz, ny, nx)`` at mass points.

    ``z = (phb + php)/g`` on full levels, averaged to half levels -- the
    same tree ``woof.core.state._height_half_from_phb`` uses for the
    base state, extended to the perturbed column.
    """
    php = state.php
    phb = state.phb
    if getattr(phb, "ndim", 0) == 1:
        phb = phb[:, None, None]
    z_full = (phb + php) / np.float32(c.G)
    return 0.5 * (z_full[:-1] + z_full[1:])


def beam_geometry(geometry: GridGeometry, site: RadarSite, *,
                  effective_earth_ratio: float = EFFECTIVE_EARTH_RATIO
                  ) -> BeamGeometry:
    """Straight-ray 4/3-earth beam geometry at every mass point.

    In the effective-earth model the ray is straight and the earth radius
    is inflated to ``k*a`` (k = 4/3 for a standard atmosphere), which
    reproduces the ray's true curvature relative to the surface.  The
    physical ground arc ``s`` between radar and target is unchanged, so
    the central angle subtended in the effective frame is ``s/(k*a)``.

    With the earth centre at the origin, the radar at radius
    ``R0 = k*a + alt`` and the target at ``R1 = k*a + z`` separated by
    central angle ``theta``, the chord and the direction cosines at the
    TARGET follow exactly:

        r        = sqrt(R0^2 + R1^2 - 2*R0*R1*cos(theta))
        sin(el)  = (R1 - R0*cos(theta)) / r
        cos(el)  =  R0*sin(theta) / r

    Both are exact (not one derived from the other by a square root), so
    the pair stays a unit vector to round-off even at grazing incidence.

    Azimuth at the target is the forward azimuth of the radar->target
    great circle evaluated at the target, i.e. the reverse of the
    target->radar azimuth.  Directly over the radar ``cos(el)`` is zero
    and azimuth is irrelevant; the code returns the (0, 1) placeholder
    there rather than a NaN, because it is multiplied by zero.

    A target coincident with the radar has no beam direction at all;
    those cells come back NaN.
    """
    if not np.isfinite(effective_earth_ratio) or effective_earth_ratio <= 0.0:
        raise ValueError(
            f"effective_earth_ratio must be positive and finite, "
            f"got {effective_earth_ratio}")

    xp_h = _array_module(geometry.height_m)
    lat = np.deg2rad(_asarray64(_array_module(geometry.latitude_deg),
                                geometry.latitude_deg))
    lon = np.deg2rad(_asarray64(_array_module(geometry.longitude_deg),
                                geometry.longitude_deg))
    height = _asarray64(xp_h, geometry.height_m)

    lat0 = np.deg2rad(np.float64(site.latitude_deg))
    lon0 = np.deg2rad(np.float64(site.longitude_deg))

    # Great-circle central angle on the REAL earth (haversine: accurate
    # for the small separations a radar actually sees).
    dlat = lat - lat0
    dlon = lon - lon0
    hav = (np.sin(0.5 * dlat) ** 2
           + np.cos(lat0) * np.cos(lat) * np.sin(0.5 * dlon) ** 2)
    gamma = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    ground_range = EARTH_RADIUS_M * gamma

    # Same ground arc, inflated radius -> smaller central angle.
    ae = effective_earth_ratio * EARTH_RADIUS_M
    theta = gamma / effective_earth_ratio
    cos_t = np.cos(theta)[None, :, :]
    sin_t = np.sin(theta)[None, :, :]

    r0 = ae + np.float64(site.altitude_m)
    r1 = ae + height
    slant = np.sqrt(np.maximum(r0 * r0 + r1 * r1 - 2.0 * r0 * r1 * cos_t, 0.0))

    coincident = slant <= 0.0
    safe = np.where(coincident, 1.0, slant)
    sin_el = np.where(coincident, np.nan, (r1 - r0 * cos_t) / safe)
    cos_el = np.where(coincident, np.nan, (r0 * sin_t) / safe)

    # Azimuth at the target of the outgoing beam = reverse of the
    # target->radar great-circle azimuth.
    y_ba = np.sin(lon0 - lon) * np.cos(lat0)
    x_ba = (np.cos(lat) * np.sin(lat0)
            - np.sin(lat) * np.cos(lat0) * np.cos(lon0 - lon))
    norm = np.hypot(y_ba, x_ba)
    overhead = norm <= 0.0
    safe_norm = np.where(overhead, 1.0, norm)
    sin_az = np.where(overhead, 0.0, -y_ba / safe_norm)
    cos_az = np.where(overhead, 1.0, -x_ba / safe_norm)

    return BeamGeometry(
        sin_elevation=sin_el, cos_elevation=cos_el,
        sin_azimuth=sin_az[None, :, :], cos_azimuth=cos_az[None, :, :],
        slant_range_m=slant, ground_range_m=ground_range,
        effective_earth_radius_m=float(ae))


# --------------------------------------------------------------------------
# winds: destaggering and earth-relative rotation
# --------------------------------------------------------------------------

def destagger_u(u):
    """``(nz, ny, nx+1)`` u faces -> ``(nz, ny, nx)`` mass points."""
    if u.ndim != 3:
        raise ValueError(f"u must be 3-D, got {u.shape}")
    return 0.5 * (u[:, :, :-1] + u[:, :, 1:])


def destagger_v(v):
    """``(nz, ny+1, nx)`` v faces -> ``(nz, ny, nx)`` mass points."""
    if v.ndim != 3:
        raise ValueError(f"v must be 3-D, got {v.shape}")
    return 0.5 * (v[:, :-1, :] + v[:, 1:, :])


def destagger_w(w):
    """``(nz+1, ny, nx)`` w faces -> ``(nz, ny, nx)`` mass points."""
    if w.ndim != 3:
        raise ValueError(f"w must be 3-D, got {w.shape}")
    return 0.5 * (w[:-1, :, :] + w[1:, :, :])


def earth_relative_winds(u_mass, v_mass, sina, cosa):
    """Rotate grid-relative mass-point winds to earth-relative.

    ``SINALPHA``/``COSALPHA`` follow the geo_em convention this repo
    transcribes from WPS ``get_rotang`` (woof/static/projection.py:21-30):
    ``alpha = wrap(stand_lon - lon) * cone``.  With true north rotated
    counter-clockwise from grid north by ``beta = -alpha``,

        u_earth = u_grid*COSALPHA - v_grid*SINALPHA
        v_earth = u_grid*SINALPHA + v_grid*COSALPHA

    which is the inverse of the rotation WRF's ``uvmet`` diagnostic
    applies, and the identity when ``sina = 0, cosa = 1`` (the idealized
    default ``DomainState`` installs, woof/core/state.py:528-529).
    """
    if sina is None and cosa is None:
        return u_mass, v_mass
    if sina is None or cosa is None:
        raise ValueError("sina and cosa must be supplied together")
    if getattr(sina, "ndim", 0) == 2:
        sina = sina[None, :, :]
    if getattr(cosa, "ndim", 0) == 2:
        cosa = cosa[None, :, :]
    return (u_mass * cosa - v_mass * sina,
            u_mass * sina + v_mass * cosa)


def mass_point_winds(state, *, rotate_to_earth: bool = True):
    """``(u_east, v_north, w)`` on mass points, ``(nz, ny, nx)`` each.

    Destaggers all three components and, unless ``rotate_to_earth`` is
    False, rotates the horizontal pair out of grid-relative coordinates
    using the state's own ``sina``/``cosa``.
    """
    u_mass = destagger_u(state.u)
    v_mass = destagger_v(state.v)
    w_mass = destagger_w(state.w)
    if rotate_to_earth:
        sina = getattr(state, "sina", None)
        cosa = getattr(state, "cosa", None)
        if sina is not None and cosa is not None:
            u_mass, v_mass = earth_relative_winds(u_mass, v_mass, sina, cosa)
    return u_mass, v_mass, w_mass


# --------------------------------------------------------------------------
# reflectivity: adapter over the product-side authority
# --------------------------------------------------------------------------

#: Schemes whose H_Z(x) has a float64 column mirror on the host
#: (:func:`_host_reflectivity`).  Every other routed scheme is answered on
#: the CUDA path only, so a caller holding host arrays has to move them to
#: the device for those (woof.da.radar_assimilation.
#: scheme_reflectivity_provider does).
HOST_REFLECTIVITY_MIRRORS = frozenset({1, 6, 10})


def _host_reflectivity(state, cfg, temperature, pressure):
    """Column-by-column float64 mirror path (host states only).

    Reference fallback, not a performance path: it makes one Python call
    per column into the certified ``woof.verify.npref`` mirrors.
    """
    from woof.verify import npref

    nz, ny, nx = np.asarray(state.p).shape
    if temperature is None:
        thb = state.thb
        if getattr(thb, "ndim", 0) == 1:
            thb = thb[:, None, None]
        pressure = state.p
        temperature = ((thb + state.thp)
                       * (np.asarray(pressure) / np.float32(c.P0))
                       ** np.float32(c.RCP))

    t = np.asarray(temperature, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    qv = np.asarray(state.qv, dtype=np.float64)

    def field(name):
        value = getattr(state, name, None)
        if value is None:
            raise ValueError(
                f"mp_physics={cfg.mp_physics} reflectivity lacks {name}")
        return np.asarray(value, dtype=np.float64)

    out = np.empty((nz, ny, nx), dtype=np.float32)
    mp = int(cfg.mp_physics)
    if mp == 10:
        qr, nr = field("qr"), field("nr")
        qs, ns = field("qs"), field("ns")
        qg, ng = field("qg"), field("ng")
        rimed = int(getattr(cfg, "morr_rimed_ice", 1))
        for j in range(ny):
            for i in range(nx):
                out[:, j, i] = npref.np_refl10cm_morrison_column(
                    qv[:, j, i], qr[:, j, i], nr[:, j, i],
                    qs[:, j, i], ns[:, j, i], qg[:, j, i], ng[:, j, i],
                    t[:, j, i], p[:, j, i], morr_rimed_ice=rimed)
    elif mp in (6, 106):
        qr, qs, qg = field("qr"), field("qs"), field("qg")
        hail = int(getattr(cfg, "wsm6_hail_opt", 0)) if mp == 6 else 0
        for j in range(ny):
            for i in range(nx):
                out[:, j, i] = npref.np_refl10cm_wsm6_column(
                    qv[:, j, i], qr[:, j, i], qs[:, j, i], qg[:, j, i],
                    t[:, j, i], p[:, j, i], hail_opt=hail)
    elif mp == 3:
        from woof.core.refl import wsm3_precipitation_phases
        rain, snow = wsm3_precipitation_phases(
            np, np.asarray(state.qr, np.float32), np.asarray(t, np.float32))
        rain = rain.astype(np.float64)
        snow = snow.astype(np.float64)
        no_graupel = np.zeros_like(rain)
        for j in range(ny):
            for i in range(nx):
                out[:, j, i] = npref.np_refl10cm_wsm6_column(
                    qv[:, j, i], rain[:, j, i], snow[:, j, i],
                    no_graupel[:, j, i], t[:, j, i], p[:, j, i], hail_opt=0)
    elif mp == 1:
        qr = field("qr")
        for j in range(ny):
            for i in range(nx):
                out[:, j, i] = npref.np_refl10cm_kessler_column(
                    qv[:, j, i], qr[:, j, i], t[:, j, i], p[:, j, i])
    elif mp == 8:
        raise NotImplementedError(
            "the host reflectivity fallback covers mp_physics 1, 6 and 10; "
            "mp_physics=8 (classic Thompson calc_refl10cm) has no float64 "
            "column mirror in woof.verify.npref, so it is only available "
            "on the CUDA path")
    elif mp == 18:
        raise NotImplementedError(
            "the host reflectivity fallback covers mp_physics 1, 6 and 10; "
            "mp_physics=18 (NSSL radardd02) has no float64 column mirror in "
            "woof.verify.npref -- unlike the three mirrored schemes it "
            "reads five ice categories, five number moments and two volume "
            "moments, and porting it is a scheme mirror rather than an "
            "adapter -- so NSSL H(x) is available on the CUDA path only, "
            "through the product's own woof.core.nssl2_diagnostics")
    elif mp == 9:
        raise NotImplementedError(
            "the host reflectivity fallback covers mp_physics 1, 6 and 10; "
            "mp_physics=9 (Milbrandt-Yau) has no float64 column mirror in "
            "woof.verify.npref -- it reads six categories and their six "
            "number moments through the scheme's own Zet block -- so "
            "Milbrandt-Yau H(x) is available on the CUDA path only, "
            "through woof.core.milbrandt2.reflectivity.  This says "
            "nothing about the scheme: mp=9 has a floor "
            "(CLEAR_AIR_FLOOR_DBZ) and a device operator, and a CuPy state "
            "is answered")
    elif mp in (14, 16, 26):
        label = {14: "WDM5", 16: "WDM6", 26: "WDM7"}[mp]
        raise NotImplementedError(
            "the host reflectivity fallback covers mp_physics 1, 6 and 10; "
            f"mp_physics={mp} ({label}) has no float64 column mirror in "
            "woof.verify.npref, so it is only available on the CUDA path, "
            "through woof.core.refl.launch_refl10cm_wdm6")
    else:
        raise NotImplementedError(
            f"mp_physics={mp} has no reflectivity formulation here; the "
            "supported set is 1 (Kessler fallback), 6 (WSM6), 8 (classic "
            "Thompson, CUDA only), 9 (Milbrandt-Yau, CUDA only), "
            "10 (Morrison), 16 (WDM6, CUDA only) and 18 (NSSL, CUDA only)")
    return out


from woof.core.state_reflectivity import (
    _nssl_reflectivity, _thompson_graupel_number, thompson_generation,
)


#: Shipped ``mp_physics`` selectors whose reflectivity is NATIVE to the
#: scheme and is NOT separable from the scheme's own state update, with the
#: reason each one is refused.  Recorded in the pattern
#: ``woof.core.microphysics_transition``'s named-refusal tables use for
#: nest edges: an exclusion that SPEAKS, so the next reader finds a
#: decision instead of an omission.
#:
#: mp=18 is deliberately NOT here.  NSSL's ``radardd02`` is a separate,
#: pure Fortran diagnostic, so it earns the real arm above; the criterion
#: is separability, not "the scheme is not in the 1/6/8/10/16/28 family".
#:
#: Without this table mp=50 fell off the end of the dispatch into
#: ``woof.core.refl.compute_refl_10cm``, whose gate answers a P3 state
#: with "do_radar_ref needs an active microphysics scheme (mp_physics 1,
#: 6, 8, 10, 16, or 28), got 50" -- false twice over -- and the host
#: fallback answered it with the same sentence it gives mp_physics=99, a
#: selector that does not exist.  A shipped scheme and a typo read
#: identically, which is the reachability defect, not the missing Z.
#:
#: mp=9 is deliberately NOT here either, though it was for a while: at
#: audit R-050 Milbrandt-Yau's Z was written by the same kernel that
#: clips ``Q >= 0`` and converts the six number moments back to #/kg, so
#: running it as H(x) would have moved the background.  The Z block has
#: since been lifted into its own kernel
#: (woof/core/kernels/milbrandt2_zet.cu), which updates nothing, and
#: ``woof.core.milbrandt2.reflectivity`` is its scheme-diagnostic H(x).
#:
#: mp=50 is no longer here either.  P3's Z was fused into p3_main's final
#: diagnostics loop (WRF module_mp_p3.F:4722-4895), which also dumps
#: sub-QSMALL condensate into vapour and theta and clamps nr and ni before
#: Z reads them; the Z half of that loop now runs on LOCAL copies in
#: woof/core/kernels/p3_zdiag.cu over the forecast port's own device
#: helpers, which updates nothing, and ``woof.core.p3_device.reflectivity``
#: is its scheme-diagnostic H(x).  The table stays, empty, for the class.
NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP: dict[int, str] = {
    5: (
        "mp_physics=5: Ferrier-Aligo forms its reflectivity inside EGCP01COLUMN from "
        "quantities that exist only in the middle of the column update: the "
        "rain intercept N0r and mean diameter INDEXR2 of the rain_pass "
        "iteration (module_mp_fer_hires.F:2292-2345), the updated rime "
        "factor, the mean ice diameter INDEXS and the large-ice fraction "
        "FLIMASS of the ice block (:2232-2251), and the falling rain and "
        "snow fluxes from the levels above.  Recomputing them from a state "
        "means running the scheme, which moves the background; the forecast "
        "REFL_10CM (the scheme's own DBZ_col, stashed by "
        "woof/core/fer_hires.py) is the field, and woof.da.hotstart "
        "accepts it through simulated_dbz=."),
    55: ("mp_physics=55: ISHMAEL forms its reflectivity inside the column "
         "update -- the rain term at module_mp_jensen_ishmael.F:2094-2095 and "
         "the three ice categories' reflectivity moments from its "
         "sedimentation pass at :2646-2649 -- so the native REFL_10CM is a "
         "product of the step.  woof.core.ishmael.reflectivity reads the "
         "state without advancing it, but this public radar operator has no "
         "bound route or clear-air floor for it yet; use the "
         "forecast-produced REFL_10CM."),
    7: ("mp_physics=7: NU-WRF DBZ is formed inside saticel_s "
        "(module_mp_gsfcgce_4ice_nuwrf.F:4313-4317, written to refl_10cm at "
        ":4498) from process-local slopes and intercept factors after its "
        "state update; replaying that update as H(x) would change the "
        "analyzed moisture. The standalone refl10cm_gsfc (:4781-4950) is not "
        "the mp7 active diagnostic. Use the forecast-produced REFL_10CM."),
}
NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP[15] = NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP[5].replace("mp_physics=5:", "mp_physics=15:")

#: Schemes whose H_Z(x) is the scheme's own state-reading diagnostic and
#: which take BOTH a device and a host state themselves, dispatched ahead
#: of the host/device split.  A row, not a branch: ``module:function``.
STATE_REFLECTIVITY_OPERATORS: dict[int, str] = {
    50: "woof.core.p3_device:reflectivity",
    51: "woof.core.p3_device:reflectivity",
    52: "woof.core.p3_device:reflectivity",
    53: "woof.core.p3_device:reflectivity",
    56: "woof.core.ntu:reflectivity",
}


from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
# Named schemes compute Z inside their column call in this port; a pure
# H(x) entry is not wired yet, so the radar operator refuses them by name.
#: Where each named scheme's own reflectivity is defined, at its pin.
_NAMED_Z_AUTHORITY = {
    "tempo": "NCAR/TEMPO 9adf9ef module_mp_tempo_diags.F90:59-121 (reflectivity_10cm)",
}
NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP.update({
    _s.mp_id: (f"mp_physics={_s.mp_id} ({_s.label}): Z is computed inside the scheme's column call "
               f"in this port ({_NAMED_Z_AUTHORITY[_s.name]}); no standalone H(x) entry is wired yet")
    for _s in _NAMED_MP_SCHEMES.values() if _s.native_reflectivity})

def _refuse_unrouted_reflectivity(mp_physics: int) -> None:
    """Name a shipped scheme whose H(x) route nobody has written.

    Audit R-050.  ``NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP`` covers the
    schemes whose native Z is entangled with their own state update.  A
    scheme can also be shipped, produce REFL_10CM, and simply have no
    route here yet -- mp=9 was exactly that -- and until this existed it
    fell through to ``_host_reflectivity``'s final ``else``, which answers
    a shipped scheme with the sentence it gives ``mp_physics=99``: "has no
    reflectivity formulation here".  A shipped scheme and a typo read
    identically, which is the reachability defect the tables above exist
    to stop, so the registry's own ``radar_da.reflectivity_route`` row is
    read here, before the host/device split, and its recorded reason is
    the refusal.

    Silent for every routed scheme and for a selector the registry does
    not implement: the first is not a refusal and the second is already
    answered, correctly, by the dispatch below.
    """

    from woof.physics_registry import consumer_row_for_selector

    row = consumer_row_for_selector("microphysics", "radar_da", mp_physics)
    if not isinstance(row, dict) or row.get("reflectivity_route") != "unrouted":
        return
    reason = row.get("reflectivity_route_reason") or (
        "woof/physics_registry_v2.json records no route for it")
    raise NotImplementedError(
        f"mp_physics={mp_physics} has no H(x) route for reflectivity: "
        f"{reason}.  Supply an independently verified field through "
        "woof.da.hotstart simulated_dbz=, or assimilate velocity only "
        "for this scheme.")


def simulated_reflectivity(state, cfg, *, temperature=None, pressure=None,
                           thompson_graupel_number=None):
    """H_Z(x): 10 cm reflectivity ``(nz, ny, nx)`` in dBZ.

    A shape-checking adapter over the product-side authority; see the
    module docstring for which formulation each ``cfg.mp_physics`` binds
    to and what PSD assumptions it embeds.  No Z math lives here.

    ``cfg`` is required, and positional, exactly as it is for
    ``woof.core.refl.compute_refl_10cm(state, cfg, ...)``: the operator
    is only meaningful once the active microphysics scheme is known, and
    ``DomainState`` does not carry its own ``RunConfig``.

    ``temperature``/``pressure`` are the microphysics-time pair WRF hands
    the diagnostic (``t1d``/``p1d``); supply both or neither.  Omitting
    them diagnoses temperature from the current state, which is the
    right choice for a DA operator evaluated between steps.

    ``mp_physics`` 8 and 28 carry no graupel number in the Registry and
    their Z reads one, so this operator DERIVES it from the state with
    the scheme's own wrapper diagnosis when the caller supplies none
    (:func:`_thompson_graupel_number`).  The product-side authority
    requires the moment instead, because its caller is inside the
    microphysics call and has the evolved one; an operator evaluated
    between steps has no evolved one to have, and refusing there left
    the whole Thompson family with no H_Z(x) at all.

    ``mp_physics=50`` dispatches through
    :data:`STATE_REFLECTIVITY_OPERATORS` to
    ``woof.core.p3_device.reflectivity``, the Z half of P3's own final
    diagnostics replayed on local copies (device kernel, host reference),
    before the host/device split.  It is not routed to
    ``compute_refl_10cm``, whose only P3-satisfiable branch is the
    rain-only Kessler fallback.

    ``mp_physics=18`` dispatches to the product's NSSL ``radardd02``
    diagnostic exactly as the production coordinator does (CUDA only, no
    host mirror; see :func:`_nssl_reflectivity` for the two caveats a DA
    caller has to carry -- a 0 dBZ clear-air floor rather than -35, and the
    per-mass moment convention).  It is a supported scheme, not mapped onto
    Morrison or Thompson.

    On a CuPy state this returns the state's persistent ``refl_10cm``
    scratch slot, owned by the state and reused on the next call --
    copy it if you need to keep it.  On a NumPy state it returns a fresh
    float32 array.
    """
    if state.qv is None:
        raise ValueError("simulated reflectivity requires a moist state")
    if (temperature is None) != (pressure is None):
        raise ValueError("temperature and pressure must be supplied together")

    # Ahead of the host/device split, because neither path has this scheme's
    # Z and both used to answer a shipped scheme the way they answer a
    # selector that does not exist.
    refusal = NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP.get(int(cfg.mp_physics))
    if refusal is not None:
        raise NotImplementedError(refusal)
    operator = STATE_REFLECTIVITY_OPERATORS.get(int(cfg.mp_physics))
    if operator is not None:
        import importlib

        module_name, function_name = operator.split(":")
        return getattr(importlib.import_module(module_name), function_name)(
            state, temperature=temperature, pressure=pressure)
    from woof.core.mp_column_schemes import MP_COLUMN_SCHEMES
    if int(cfg.mp_physics) in MP_COLUMN_SCHEMES:
        # A table-admitted scheme's H_Z(x) is its own refl routine on the
        # state's columns (the row names it), ahead of the host/device
        # split because the routine is device-only and says so itself.
        from woof.core.mp_column_schemes import reflectivity
        return reflectivity(state, cfg, temperature=temperature,
                            pressure=pressure)
    _refuse_unrouted_reflectivity(int(cfg.mp_physics))

    xp = _array_module(state.p)
    if xp is np:
        return _host_reflectivity(state, cfg, temperature, pressure)

    if int(cfg.mp_physics) == 18:
        if pressure is not None:
            raise ValueError(
                "the NSSL radardd02 diagnostic diagnoses its own density "
                "and reads pressure off the state; it does not take the "
                "t1d/p1d microphysics-time pair the 1/6/8/10 routes do")
        return _nssl_reflectivity(state, temperature, pressure, cfg)

    if int(cfg.mp_physics) == 9:
        # Milbrandt-Yau owns REFL_10CM natively and its Z block updates
        # nothing, so it is dispatched to the scheme's own operator exactly
        # as mp=18 is -- not mapped onto Morrison and not routed into
        # compute_refl_10cm, which carries no mp=9 branch because the
        # scheme needs none.
        from woof.core.milbrandt2 import reflectivity as my2_reflectivity

        return my2_reflectivity(
            state, temperature=temperature, pressure=pressure)

    if (int(cfg.mp_physics) == 28 and thompson_generation(cfg) == "wrf_461"
            and temperature is None):
        from woof.core.thompson_aerosol_state import launch_aerosol_exner
        pressure = state.p
        exner = xp.empty_like(pressure)
        launch_aerosol_exner(pressure, exner)
        thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
        temperature = state.scratch(tuple(pressure.shape), "refl_t")
        temperature[...] = (thb + state.thp) * exner

    if (int(cfg.mp_physics) in (8, 28)
            and thompson_graupel_number is None):
        temperature, pressure, thompson_graupel_number = (
            _thompson_graupel_number(state, temperature, pressure,
                                     version=thompson_generation(cfg),
                                     aerosol=int(cfg.mp_physics) == 28))

    from woof.core.refl import compute_refl_10cm

    return compute_refl_10cm(
        state, cfg, temperature=temperature, pressure=pressure,
        thompson_graupel_number=thompson_graupel_number)


# --------------------------------------------------------------------------
# fall speed
# --------------------------------------------------------------------------

#: The precipitating species the fall-speed gate reads, in the order the
#: reflectivity authority reads them.  Cloud water and cloud ice are
#: excluded: they do not sediment at radar fall speeds, and the bound Z
#: formulations read no cloud moment.
PRECIPITATING_SPECIES = ("qr", "qs", "qg", "qh")


def precipitating_mixing_ratio(state):
    """Total precipitating water ``(nz, ny, nx)``: qr + qs + qg (+ qh).

    A DIAGNOSTIC, and no longer the activity gate.  Summing the species
    and thresholding the sum is not what the reflectivity authority does
    and not what this module claimed: see
    :func:`precipitating_activity_mask`.
    """
    total = None
    for name in PRECIPITATING_SPECIES:
        value = getattr(state, name, None)
        if value is None:
            continue
        total = value if total is None else total + value
    if total is None:
        raise ValueError(
            "no precipitating hydrometeor fields (qr/qs/qg/qh) on the state; "
            "pass fall_speed='none' for an air-motion-only radial velocity")
    return total


def precipitating_activity_mask(state):
    """Where ANY precipitating species is active: ``q_species > 1e-9``.

    PER SPECIES, which is what the reflectivity authority this module is
    bound to actually does.  ``refl10cm_hm``
    (module_mp_morr_two_moment.F:4502-4675) and every woof mirror of it
    test each category against ``1e-9 kg/kg`` on its own and contribute
    that category's Ze only where its own test passes; a category below
    the threshold contributes nothing, whatever the other categories are
    carrying.

    Summing ``qr+qs+qg(+qh)`` and thresholding the SUM -- which is what
    this gate did -- is a different rule, and it differs exactly where it
    matters.  Four species at ``3e-10 kg/kg`` each are individually
    inactive by the authority's rule and contribute no reflectivity at
    all, but their ``1.2e-9`` sum cleared the summed gate and switched on
    a 5.244 m/s fall speed in a cell the Z operator calls empty.  A
    Doppler operator that sediments hydrometeors the reflectivity
    operator says are not there is inconsistent with its own H_Z, which is
    the one consistency this module is for.

    The other candidate authority is WRFDA's Sun-Crook implementation,
    which gates on RAIN ONLY above ``1e-5 kg/kg``.  That is a defensible
    rule and it is NOT the one implemented here: this operator's Z comes
    from the model's own multi-species scheme, so gating it on rain alone
    would switch the fall speed off in snow and graupel the same Z
    operator is reporting.  The authority chosen is the reflectivity
    authority, because it is the field being consumed.
    """
    xp = None
    mask = None
    for name in PRECIPITATING_SPECIES:
        value = getattr(state, name, None)
        if value is None:
            continue
        if xp is None:
            xp = _array_module(value)
        active = value > Q_ACTIVE_THRESHOLD
        mask = active if mask is None else (mask | active)
    if mask is None:
        raise ValueError(
            "no precipitating hydrometeor fields (qr/qs/qg/qh) on the state; "
            "pass fall_speed='none' for an air-motion-only radial velocity")
    return mask


def frozen_mass_weighted_fall_speed(name, q, density):
    """Lin et al. (1983) mass-weighted fall speed (m/s) of one frozen
    species with mixing ratio ``q`` (kg/kg) in air of ``density`` (kg/m3).

    For an exponential distribution ``N0 exp(-lambda D)`` of spheres of
    bulk density ``rho_x`` falling at ``a D^b (RHO0/rho)^0.5``,
    ``lambda = (pi rho_x N0 / (rho q))**0.25`` and the mass-weighted mean
    is ``a Gamma(4+b) / (6 lambda^b) (RHO0/rho)^0.5``.  Zero where ``q``
    is not positive.
    """
    from math import gamma

    xp = _array_module(q)
    a, b, n0, bulk = FROZEN_FALL_SPEED_CONSTANTS[name]
    q = xp.asarray(q, dtype=xp.float64)
    rho = xp.asarray(density, dtype=xp.float64)
    positive = q > 0.0
    safe_q = xp.where(positive, q, 1.0)
    slope = (np.pi * bulk * n0 / (rho * safe_q)) ** 0.25
    vt = (a * gamma(4.0 + b) / 6.0) * slope ** (-b) * (
        FALL_SPEED_RHO0 / rho) ** 0.5
    return xp.where(positive, vt, 0.0)


def species_blended_fall_speed(rain_vt, species, inverse_density, *,
                               max_ms: float = FALL_SPEED_MAX_MS):
    """The cell's fall speed as the mass-weighted blend of its ACTIVE
    precipitating species: rain at ``rain_vt`` (Sun and Crook, from the
    scheme's Z), snow, graupel and hail at
    :func:`frozen_mass_weighted_fall_speed`.

    ``species`` maps ``qr``/``qs``/``qg``/``qh`` (any subset) to mixing
    ratios; a species is active above :data:`Q_ACTIVE_THRESHOLD`, per
    species, as the reflectivity authority tests it.  Where no species is
    active the result is ``rain_vt`` (the caller's activity gate then
    zeroes it).  Where rain is the only active species the result is
    ``rain_vt`` exactly, so a rain-only column is unchanged.
    """
    xp = _array_module(rain_vt)
    density = 1.0 / xp.asarray(inverse_density, dtype=xp.float64)
    rain = xp.asarray(rain_vt, dtype=xp.float64)
    weight_total = xp.zeros_like(rain)
    weighted = xp.zeros_like(rain)
    frozen_active = xp.zeros(rain.shape, dtype=bool)
    for name in PRECIPITATING_SPECIES:
        q = species.get(name)
        if q is None:
            continue
        q = xp.asarray(q, dtype=xp.float64)
        active = q > Q_ACTIVE_THRESHOLD
        mass = xp.where(active, q, 0.0)
        if name == "qr":
            speed = rain
        else:
            speed = frozen_mass_weighted_fall_speed(name, mass, density)
            frozen_active = frozen_active | active
        weight_total = weight_total + mass
        weighted = weighted + mass * speed
    blended = xp.where(weight_total > 0.0,
                       weighted / xp.where(weight_total > 0.0,
                                           weight_total, 1.0), rain)
    blended = xp.clip(blended, 0.0, max_ms)
    # A cell with no active frozen species keeps the rain value bit for bit.
    return xp.where(frozen_active, blended, rain).astype(
        rain_vt.dtype, copy=False)


def reflectivity_fall_speed(dbz, pressure, active=None, *,
                            surface_pressure=None,
                            max_ms: float = FALL_SPEED_MAX_MS,
                            species=None, inverse_density=None):
    """Sun and Crook reflectivity-driven fall speed (m/s, downward).

    ``surface_pressure`` is the ``(ny, nx)`` pressure at the ground, in Pa,
    and it is REQUIRED.  Sun and Crook's density factor is ``(ps/p)**0.4``
    with ``ps`` the surface pressure, exactly as WRFDA's
    ``da_radial_velocity.inc`` takes it -- a separate ``ps`` argument, then
    ``alpha = (ps/p)**0.4``.  Substituting the thermodynamic reference
    constant ``P0 = 1000 hPa`` is a different closure and a systematic one:
    the ratio of the two is ``(100000/ps)**0.4``, constant through the
    column, so it is +4.3% at a 900 hPa surface, +9.3% at 800 hPa and
    +15.3% at 700 hPa, always making ``vt`` too large and always pushing
    the positive-away radial velocity inbound by ``-dvt*sin(el)``.  There
    is no default because a constant is not a field.

    Sun and Crook state two relations for Marshall-Palmer rain, both with
    ``rho*qr`` in g m-3 and ``a = (ps/p)**0.4``:

        Z[dBZ] = 43.1 + 17.5*log10(rho*qr)
        vt     = 5.40 * a * (rho*qr)**0.125

    Eliminating ``rho*qr`` gives the form used here, which consumes the
    bound Z operator's output directly:

        vt = 5.40 * (ps/p)**0.4 * 10**(0.125*(Z - 43.1)/17.5)

    The Z leg is not an outside assumption: it *is* the Smith et al.
    (1975) exponential-PSD form woof's Kessler fallback implements
    (woof/core/refl.py:30-43).  At ``rho*qr = 1 g m-3`` that PSD gives
    ``lambda = (pi*1000*8e6/1e-3)**0.25 = 2239 m-1`` and
    ``Ze = 720*8e6/lambda**7 * 1e18 = 2.04e4 mm^6 m^-3`` = 43.1 dBZ,
    exactly Sun and Crook's intercept.  So on the ``mp_physics=1`` path
    the fall speed and the reflectivity share one PSD; on the Morrison,
    WSM6 and Thompson paths their richer PSDs supply Z while the fall
    speed is still read off the Marshall-Palmer rain calibration, which
    is the documented approximation (see the module docstring).

    Because the relation is a power law it does not vanish in clear air
    (~1.7 m/s at the -35 dBZ floor), so ``active`` gates it to cells the
    reflectivity authority calls active.  ``active`` is a BOOLEAN mask,
    normally :func:`precipitating_activity_mask` -- ANY species above
    ``1e-9 kg/kg``, tested per species exactly as ``refl10cm_hm`` tests
    them.  A float mixing-ratio array is REFUSED rather than thresholded:
    this argument used to be the summed ``qr+qs+qg(+qh)`` field, gated at
    ``1e-9``, and four species at ``3e-10`` each -- every one of them
    inactive to the Z operator -- summed past it and switched on a
    5.244 m/s fall speed.  Accepting the old argument silently would keep
    that behaviour reachable from every existing caller.

    Non-finite dBZ (Morrison's invalid-number-moment signal) propagates as
    non-finite fall speed rather than being silently zeroed.

    ``species`` (``{"qr", "qs", "qg", "qh": mixing ratio}``, any subset)
    with ``inverse_density`` (``alt``, m3/kg) switches on the per-species
    blend (:func:`species_blended_fall_speed`): frozen species fall at
    their own Lin et al. (1983) speeds instead of the rain relation.
    Both callers in this tree pass them; without them the rain relation
    applies to every species, as it always did.
    """
    xp = _array_module(dbz)
    if (species is None) != (inverse_density is None):
        raise ValueError(
            "species= and inverse_density= go together: the frozen "
            "species' fall speeds need the air density")
    if max_ms <= 0.0 or not np.isfinite(max_ms):
        raise ValueError(f"max_ms must be positive and finite, got {max_ms}")
    if surface_pressure is None:
        raise ValueError(
            "reflectivity_fall_speed needs surface_pressure (Pa, (ny, nx)): "
            "Sun and Crook's density factor is (ps/p)**0.4 with ps the "
            "pressure at the ground, not the 1000 hPa thermodynamic "
            "reference.  Substituting the constant is a +4.3% error in vt "
            "over a 900 hPa surface and +15.3% over a 700 hPa one, always "
            "in the same direction, so it is not something to default")

    ps = xp.asarray(surface_pressure)
    if ps.ndim == 2:
        ps = ps[None, :, :]
    elif ps.ndim != 3:
        raise ValueError(
            f"surface_pressure must be (ny, nx) or (nz, ny, nx), got "
            f"shape {tuple(ps.shape)}")
    expected = tuple(np.asarray(_to_host(xp, dbz)).shape)[1:]
    if tuple(np.asarray(_to_host(xp, ps)).shape)[1:] != expected:
        raise ValueError(
            f"surface_pressure {tuple(np.asarray(_to_host(xp, ps)).shape)} "
            f"does not sit on the {expected} horizontal grid the "
            "reflectivity does")
    density_factor = xp.power(ps.astype(dbz.dtype, copy=False) / pressure,
                              FALL_SPEED_DENSITY_EXPONENT)
    exponent = (SUN_CROOK_VT_EXPONENT * (dbz - SUN_CROOK_Z_INTERCEPT_DBZ)
                / SUN_CROOK_Z_SLOPE_DB)
    vt = (SUN_CROOK_VT_COEFF * density_factor
          * xp.power(xp.asarray(10.0, dtype=dbz.dtype), exponent))
    vt = xp.clip(vt, 0.0, max_ms)
    if species is not None:
        vt = species_blended_fall_speed(vt, species, inverse_density,
                                        max_ms=max_ms)
    if active is not None:
        mask = xp.asarray(active)
        if mask.dtype != bool:
            raise ValueError(
                "the fall-speed activity gate takes a BOOLEAN mask, got "
                f"dtype {mask.dtype}. Build it with "
                "woof.da.obsop.precipitating_activity_mask(state), which "
                "tests each precipitating species against "
                f"{Q_ACTIVE_THRESHOLD:g} kg/kg the way the reflectivity "
                "authority does. Passing the summed mixing ratio here "
                "gated the SUM, which activates cells whose every species "
                "is inactive")
        vt = xp.where(mask, vt, xp.zeros_like(vt))
    return vt


# --------------------------------------------------------------------------
# radial velocity
# --------------------------------------------------------------------------

def radial_velocity(state, site: RadarSite, geometry: GridGeometry = None, *,
                    cfg=None, reflectivity_dbz=None,
                    fall_speed="reflectivity", surface_pressure=None,
                    effective_earth_ratio: float = EFFECTIVE_EARTH_RATIO,
                    rotate_to_earth: bool = True,
                    fall_speed_max_ms: float = FALL_SPEED_MAX_MS,
                    dtype=None):
    """H_Vr(x): Doppler radial velocity ``(nz, ny, nx)`` in m/s.

    Positive AWAY from the radar (outbound), the standard convention.

        Vr = u_e*cos(el)*sin(az) + v_n*cos(el)*cos(az) + (w - vt)*sin(el)

    with elevation and azimuth taken at the model mass point under 4/3-
    earth beam geometry (:func:`beam_geometry`), winds destaggered to
    mass points and rotated earth-relative, and ``vt`` the downward
    hydrometeor fall speed.

    ``geometry`` may be omitted only if the state carries ``xlat``/
    ``xlong``; otherwise :meth:`GridGeometry.from_state` raises rather
    than guessing where the domain is.

    ``fall_speed`` is one of:

    - ``"reflectivity"`` (default) -- Sun and Crook (1997) from the bound
      Z operator's own dBZ.  Supply ``reflectivity_dbz=`` (cheapest, and
      what an ensemble filter already has in hand) or ``cfg=`` to let
      this call :func:`simulated_reflectivity` itself.  Supplying
      neither raises: silently dropping the fall-speed term would change
      the operator's physics without saying so.  This closure also needs
      ``surface_pressure=`` -- ``(ny, nx)`` in Pa -- for its density
      factor, on the same argument.
    - ``"none"`` -- pure air motion, ``vt = 0``.  A documented
      simplification, not a default.
    - an array ``(nz, ny, nx)`` -- caller's own closure, m/s downward.

    ``dtype`` defaults to the state's wind dtype (float32).  Pass
    ``numpy.float64`` for analytic work where round-off matters.
    """
    if geometry is None:
        geometry = GridGeometry.from_state(state)

    u_e, v_n, w_mass = mass_point_winds(state, rotate_to_earth=rotate_to_earth)
    shape = tuple(np.asarray(_to_host(_array_module(w_mass), w_mass)).shape)
    if geometry.shape != shape:
        raise ValueError(
            f"grid geometry {geometry.shape} does not match the state's "
            f"mass-point shape {shape}")

    out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
    xp = _array_module(w_mass)

    beam = beam_geometry(geometry, site,
                         effective_earth_ratio=effective_earth_ratio)
    east, north, up = beam.unit_vector_enu()
    if xp is not np:
        east = xp.asarray(east, dtype=out_dtype)
        north = xp.asarray(north, dtype=out_dtype)
        up = xp.asarray(up, dtype=out_dtype)
    else:
        east = east.astype(out_dtype, copy=False)
        north = north.astype(out_dtype, copy=False)
        up = up.astype(out_dtype, copy=False)

    vt = _resolve_fall_speed(state, cfg, reflectivity_dbz, fall_speed,
                             shape, xp, out_dtype, fall_speed_max_ms,
                             surface_pressure)

    u_e = u_e.astype(out_dtype, copy=False)
    v_n = v_n.astype(out_dtype, copy=False)
    w_mass = w_mass.astype(out_dtype, copy=False)

    vertical = w_mass if vt is None else (w_mass - vt)
    return (u_e * east + v_n * north + vertical * up).astype(out_dtype,
                                                             copy=False)


def _resolve_fall_speed(state, cfg, reflectivity_dbz, fall_speed, shape, xp,
                        out_dtype, max_ms, surface_pressure=None):
    """Turn the ``fall_speed`` argument into an array or None."""
    if fall_speed == "none":
        return None
    if fall_speed == "reflectivity":
        if reflectivity_dbz is None:
            if cfg is None:
                raise ValueError(
                    "fall_speed='reflectivity' needs the bound Z operator's "
                    "output: pass reflectivity_dbz= (preferred -- an "
                    "ensemble filter already has it) or cfg= so this call "
                    "can evaluate simulated_reflectivity itself, or ask for "
                    "fall_speed='none' to accept the air-motion-only "
                    "simplification explicitly")
            reflectivity_dbz = simulated_reflectivity(state, cfg)
        dbz = reflectivity_dbz.astype(out_dtype, copy=False)
        pressure = state.p.astype(out_dtype, copy=False)
        species = {name: getattr(state, name) for name in
                   PRECIPITATING_SPECIES
                   if getattr(state, name, None) is not None}
        alt = getattr(state, "alt", None)
        vt = reflectivity_fall_speed(
            dbz, pressure, precipitating_activity_mask(state),
            surface_pressure=surface_pressure, max_ms=max_ms,
            species=species if alt is not None else None,
            inverse_density=alt)
        return vt.astype(out_dtype, copy=False)
    if isinstance(fall_speed, str):
        raise ValueError(
            f"fall_speed must be 'reflectivity', 'none', or an array, "
            f"got {fall_speed!r}")
    vt = fall_speed
    vt_shape = tuple(np.asarray(_to_host(_array_module(vt), vt)).shape)
    if vt_shape != shape:
        raise ValueError(
            f"fall_speed array {vt_shape} must match the mass-point "
            f"shape {shape}")
    return vt.astype(out_dtype, copy=False)


# ---------------------------------------------------------------------------
# AGREEMENT WITH THE REGISTRY, AT IMPORT.  The three tables above are the
# source of every option's ``consumers.radar_da`` row
# (tools/build_registry.py); this holds them to the generated copy, so a
# scheme that gains a floor here without a rebuilt registry, or a registry
# claiming a floor this module does not carry, fails this import rather
# than the first analysis.  A selector in none of the three tables is the
# registry's "unread" status, and that is a decision the registry states
# (audit R-015 for mp=16), not an omission.  mp=9 carries an unread FLOOR
# (R-015, still open) and a recorded ROUTE (R-050, closed), which is why
# it appears here with a route and no number.
def _require_agreement_with_the_registry() -> None:
    from woof.physics_registry import require_consumer_rows_agreement

    observed: dict[int, dict] = {}
    for mp, floor in CLEAR_AIR_FLOOR_DBZ.items():
        observed[mp] = {"status": "one-number", "floor": float(floor)}
    for mp in CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER:
        observed[mp] = {"status": "not-one-number", "floor": None}
    for mp in NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP:
        observed.setdefault(mp, {"status": "unread", "floor": None})
        observed[mp]["route"] = "native-not-separable"

    def project(row):
        status = row["clear_air_floor_status"]
        if status in ("unread", "no-microphysics") \
                and row["reflectivity_route"] != "native-not-separable":
            # Neither table names the scheme: the module's silence IS the
            # registry's "unread", and there is nothing to compare.
            return None
        projected = {"status": status,
                     "floor": (None if row["clear_air_floor_dbz"] is None
                               else float(row["clear_air_floor_dbz"]))}
        if row["reflectivity_route"] == "native-not-separable":
            projected["route"] = "native-not-separable"
        return projected

    require_consumer_rows_agreement(
        "woof.da.obsop (CLEAR_AIR_FLOOR_DBZ, CLEAR_AIR_FLOOR_IS_NOT_ONE_NUMBER, "
        "NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP)",
        "microphysics", "radar_da", observed, project=project,
        cited_absences={
            0: "no microphysics, no reflectivity operator",
            11: ("CAMMGMP never writes REFL_10CM; initialized zeros are "
                 "retained by woof.core.refl, and no radar H(x) is bound"),
            30: ("FAST_SBM writes no REFL_10CM (woof.core.fast_sbm_runtime."
                 "apply refuses a due frame), so radar DA has no H(x) on "
                 "mp_physics=30 until a reflectivity operator over the bin "
                 "spectra is wired"),
            95: "WRF module_mp_etanew.F has no REFL_10CM producer or radar operator",
            # 3.0.0 port folds: each scheme keeps its own native field (or
            # none) and no standalone radar H(x) is bound for it yet.
            27: ("UDM writes REFL_10CM inside udm_step (woof.core.refl."
                 "SCHEME_NATIVE_REFL_10CM[27]); no standalone radar H(x) or "
                 "clear-air floor is bound for mp_physics=27"),
            29: ("RCON computes REFL_10CM inside rcon_step (woof.core.refl."
                 "SCHEME_NATIVE_REFL_10CM[29]); no standalone radar H(x) or "
                 "clear-air floor is bound for mp_physics=29"),
            96: ("WRF's MADWRF_MP arm is empty and never writes REFL_10CM "
                 "(woof.core.refl.SCHEME_ABSENT_REFL_10CM[96]); there is no "
                 "reflectivity to assimilate"),
        })


_require_agreement_with_the_registry()
