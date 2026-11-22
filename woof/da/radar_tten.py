"""Radar latent heating on the device: the tendency builder and the forcing.

What this is
------------
HRRR does not only analyse radar reflectivity, it heats the model with it.
A stand-alone program turns observed three-dimensional reflectivity and a
model background into a potential-temperature tendency, four times over the
hour before a forecast, and the model then uses that tendency IN PLACE OF the
microphysics heating wherever the radar has coverage.  Where the radar saw
echo the model is heated as a storm of that strength would heat it; where the
radar looked and saw nothing the tendency is zero, so latent heating is
withheld from a storm the model made up; where the radar has no coverage the
model's own microphysics heats as usual.

This module is that pair on the GPU, for the research data assimilation line:

* :func:`build_tendency` is the builder.  It follows NOAA's subroutines
  statement by statement and is graded against them compiled unchanged
  (``tools/radar_tten_oracle``).
* :class:`RadarTtenForcing` is the model side.  ``woof.core.microphysics
  .apply`` reads one optional attribute off the state; with a forcing there,
  theta takes ``tendency * dt`` in place of the microphysics increment on
  covered points and the microphysics increment everywhere else.

Source, and what "the same" means
---------------------------------
NOAA-EMC/HRRR at tag ``v4.1.21``.  File names below are relative to
``sorc/hrrr_ref2tten.fd/`` unless they name the model.

* ``pbl_height.f90``: the boundary-layer top as a fractional level index
  from virtual potential temperature, in single precision.
* ``build_missing_REFcone.f90``: fills the radar's cone of silence from a
  tabulated profile (the season is fixed to summer, ``:163``) and sets every
  level up to the boundary-layer top to no coverage (``:239-241``).
* ``radar_ref2tten.f90``: the probable-convection test (``:143-162``), the
  tendency (``:177-217``), two smoothing passes (``:219-222``), no coverage
  restored after smoothing (``:227-234``), three more passes and the column
  flag (``:236-309``), stratiform columns handed back to the model
  (``:315-328``) and the final sentinel (``:340-346``).
* ``smooth.f90``: a nine-point stencil on the unsmoothed field for the
  interior, then the four edges updated in place in sequence, so each edge
  is a recurrence along itself.
* ``gsdcloudanalysis_ref2tten.f90``: the driver whose constants are the
  defaults of :class:`RadarTtenConfig` (``:159-162``) and whose slot times
  are 15, 30, 45 and 60 minutes (``:438-441``).
* ``sorc/hrrr_wrfarw.fd/WRFV3.9/dyn_em/module_big_step_utilities_em.F``:
  the slot choice (``:5913-5938``) and the application (``:5991-6005``).

Precision matches NOAA's: inputs and the output are float32, the tendency
is float64 inside the builder, and every arithmetic node is one
round-to-nearest operation with no fused multiply-add, which is what
gfortran emits with ``-ffp-contract=off``.  The two powers go through the
tree's own transcription of glibc's binary64 ``pow``
(``woof/core/kernels/glibc_flt64.cuh``), because CUDA's is not the same
function.

Values in the reflectivity array (NOAA's convention, kept)
---------------------------------------------------------
``>= 0.001`` dBZ is echo; ``-99`` is "the radar looked and found nothing";
``-99999`` is no coverage.  :func:`reflectivity_from_document` builds that
array from a ``gpuwm-obs.radar-grid`` document on the device.

Values in a tendency slot (NOAA's convention, kept)
---------------------------------------------------
Levels below the top hold K/s, or ``-20`` for no coverage.  The TOP level
is not a tendency: it holds a two-dimensional flag, ``-10`` no information,
``0`` no convection, ``1`` convection nearby.  A slot has exactly the shape
and values of ``RAD_TTEN_DFI_1..4`` in a NOAA start file, so those four
fields attach unchanged through :meth:`RadarTtenForcing.from_fields`.

``vinterp_radar_ref.f90`` (the mosaic's fixed heights to model levels) is
ported as :func:`vinterp_mosaic` for a three-dimensional mosaic input; the
radar grid files this line reads today are already on model levels and do
not need it.

The companion clamp
-------------------
HRRR never runs ``mp_tend_radar = 1`` alone: the same namelist sets
``mp_tend_lim = 0.07`` K/s (``parm/conus/hrrr_wrfpre.nl:108-109``; the
HRRRDAS members carry the same value, ``parm/hrrrdas/hrrrdas_wrf.nl:102``).
WRF clamps the microphysics increment to ``+/- mp_tend_lim * dt`` before the
radar select (``module_big_step_utilities_em.F:5968-5969``), so the clamp
bounds the heating of every point the radar does not cover and the stored
rate ``h_diabatic`` everywhere (``:6014``), and never the radar tendency.
A forcing therefore runs its microphysics with
:data:`HRRR_MP_TEND_LIM` in place of the case's clamp, by default, and its
receipt records both values.

Where this sits against HRRR's own arrangement (a declared divergence)
----------------------------------------------------------------------
In HRRR only the deterministic pre-forecast is forced
(``parm/conus/hrrr_wrfpre.nl:109``); the HRRRDAS ensemble members run
``mp_tend_radar = 0`` (``parm/hrrrdas/hrrrdas_wrf.nl:101``) and the free
forecast does too (``parm/conus/hrrr_wrf.nl:108``).  The research cycle
(``tools/da_cycle_prepared.py --radar-tten``) has no analysed deterministic
member yet, so it forces the ENSEMBLE members, the reverse of HRRR.  Two
consequences follow and are measured, not assumed
(``tools/radar_tten_proof/spread.py``): each observation is used twice
inside the ensemble (to force the members, then by the filter at the end
of the leg against the same file), and the members receive nearly the
same heating where the radar covers, which shrinks their spread there.

WOOF options around NOAA's builder (design E3, all off by default)
----------------------------------------------------------------
* Windows: :func:`build_forcing` takes one volume per 15-minute window and
  :func:`window_end_minutes` gives NOAA's slot times for a leg; the cycle
  reads gridded Level II windows (``--radar-tten-windows``).
* Per-member parameters: :func:`member_config` draws dt_cond and the
  28 dBZ thresholds per member, deterministic in (seed, member).
* The suppression rule: :func:`suppression_receipt` checks every built
  slot.  NOAA's two smoothing passes run over observed-no-echo points too,
  so heat bleeds into air the radar saw as empty, and a stratiform column
  hands observed-clear points back to the model's own heating;
  ``strict_suppression`` makes "observed no echo" mean exactly zero
  heating.  Note also that NOAA sets warm echo under 28 dBZ to ZERO
  tendency (it replaces the model's heating there) rather than leaving the
  model alone, unless the column is handed back as stratiform.
* ``pbl_extension``: heat down to the PBL top where 200 hPa of the column
  is observed, instead of NOAA's level-7 floor.
* :class:`LatentHeatNudging`: the comparison arm, attached the same way.

A config with any option on is not NOAA's product; every receipt carries
:func:`config_record` with ``noaa_product``.

What is deliberately not here
-----------------------------
No NumPy implementation and no host fallback: without a CUDA device the
builder refuses.  A second implementation would be a second authority for
the same numbers.  The lightning proxy (``convert_lghtn2ref``) is not
ported because the tree has no lightning reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "HRRR_MP_TEND_LIM",
    "NO_COVERAGE_DBZ",
    "NO_COVERAGE_TENDENCY",
    "NO_ECHO_DBZ",
    "SLAB_ATTRIBUTE",
    "STATE_ATTRIBUTE",
    "RadarTtenConfig",
    "RadarTtenError",
    "RadarTtenForcing",
    "LatentHeatNudging",
    "LatentHeatNudgingConfig",
    "MEMBER_PERTURBATION",
    "WINDOWS_PER_HOUR",
    "attach",
    "background_from_state",
    "build_forcing",
    "build_forcing_from_documents",
    "build_nudging",
    "build_tendency",
    "config_record",
    "detach",
    "member_config",
    "read_window_host",
    "read_window_reflectivity",
    "suppression_receipt",
    "window_end_minutes",
    "pbl_height",
    "cone_fill",
    "reflectivity_from_document",
    "select_slot",
    "slab_extent",
    "slab_refusal",
    "smooth",
    "tendency_receipt",
    "vinterp_mosaic",
    "wrf_minutes",
]

#: Reflectivity where the radar has no coverage.  The driver's fill,
#: ``gsdcloudanalysis_ref2tten.f90:115-116, :333``, and the value
#: ``build_missing_REFcone.f90:240`` writes below the boundary-layer top.
NO_COVERAGE_DBZ = -99999.0

#: Reflectivity where the radar looked and found nothing
#: (``build_missing_REFcone.f90:178``: "in our case, -99 is no echo").
NO_ECHO_DBZ = -99.0

#: A tendency slot's value where the radar has no coverage
#: (``radar_ref2tten.f90:343``).
NO_COVERAGE_TENDENCY = -20.0

#: The attribute ``woof.core.microphysics.apply`` reads off the state.
STATE_ATTRIBUTE = "radar_tten_forcing"

#: The attribute a door sets on a resident rank's slab state: its ``(j0, i0)``
#: in the domain.  A slab forcing (one built with ``extent``) attaches only
#: to a state carrying the same origin, so it can never land on a
#: tile-streamed buffer that serves a different tile each sweep.
SLAB_ATTRIBUTE = "_radar_tten_slab"

#: K/s.  The microphysics heating clamp HRRR pairs with the radar forcing:
#: ``mp_tend_lim = 0.07`` beside ``mp_tend_radar = 1`` in the pre-forecast
#: (``parm/conus/hrrr_wrfpre.nl:108-109``), and the same value in the
#: HRRRDAS members (``parm/hrrrdas/hrrrdas_wrf.nl:102``).  WRF applies it to
#: the microphysics increment before the radar select
#: (``module_big_step_utilities_em.F:5968-5969``).  The case default is
#: WRF's Registry value, 10 K/s (``woof/config.py``, ``mp_tend_lim``).
HRRR_MP_TEND_LIM = 0.07

#: Clear-air regimes this module reads as "observed no echo": a regime built
#: from range-folded gates may be a storm, and reading it as clear would
#: withhold heating from it.  The list was owned by the reflectivity
#: adapter of the local data assimilation until 3.0.3 removed that adapter;
#: it is held here now, unchanged.
_CLEAR_AIR_SOURCES = ("finite_below_floor",
                      "below_threshold_and_finite_below_floor",
                      "undetect_and_finite_below_floor")


class RadarTtenError(ValueError):
    """The request cannot be honoured.  Never a warning."""


# --------------------------------------------------------------------------
# NOAA's constants
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RadarTtenConfig:
    """NOAA's constants, each citing the line it comes from.

    The first four are the driver's; the rest are literals inside
    ``radar_ref2tten.f90``.  Changing any of them leaves NOAA's product, and
    the oracle comparison no longer applies to the result.
    """

    #: Lowest model level that may be heated; the boundary-layer top raises
    #: it per column.  ``gsdcloudanalysis_ref2tten.f90:159``.
    krad_bot: float = 7.0
    #: Minutes over which the observed condensate is taken to have formed.
    #: ``gsdcloudanalysis_ref2tten.f90:160``.
    latent_heat_period_min: float = 20.0
    #: dBZ at or above which echo counts toward probable convection.
    #: ``gsdcloudanalysis_ref2tten.f90:161``.
    convection_refl_threshold_dbz: float = 28.0
    #: Hand columns with echo but no probable convection back to the model's
    #: microphysics.  ``gsdcloudanalysis_ref2tten.f90:162``.
    convection_only: bool = True
    #: Kelvin; 4 C.  ``radar_ref2tten.f90:152, :155, :186``.
    warm_temperature_k: float = 277.15
    #: Warmer than 4 C, echo below this is not heated.
    #: ``radar_ref2tten.f90:186``.
    warm_min_dbz: float = 28.0
    #: hPa of cold echo that makes a column probably convective.
    #: ``radar_ref2tten.f90:159``.
    cold_echo_depth_hpa: float = 200.0
    #: hPa of coverage a column needs before its flag leaves "no
    #: information".  ``radar_ref2tten.f90:279``.
    coverage_depth_hpa: float = 300.0
    #: K/s of smoothed tendency above which convection is "nearby".
    #: ``radar_ref2tten.f90:281``.
    nearby_tendency: float = 0.00002
    #: dBZ below which a covered point holds no echo.
    #: ``radar_ref2tten.f90:182, :184``.
    echo_floor_dbz: float = 0.001
    #: Reflectivity to condensate mass, ``10**(Z/a)/b*c``.
    #: ``radar_ref2tten.f90:203``.
    z_scale_dbz: float = 17.8
    z_divisor: float = 264083.0
    z_factor: float = 1.5
    #: K/s cap on the tendency.  ``radar_ref2tten.f90:210``.
    tendency_cap: float = 0.01
    #: Smoother weight.  ``radar_ref2tten.f90:220``.
    smooth_weight: float = 0.5
    #: Passes before the tendency is stored.  ``radar_ref2tten.f90:219-222``.
    smooth_passes_tendency: int = 2
    #: Further passes before the column flag.  ``radar_ref2tten.f90:238-242``.
    smooth_passes_flag: int = 3
    #: Slot end times in minutes.  ``gsdcloudanalysis_ref2tten.f90:438-441``.
    slot_minutes: tuple = (15.0, 30.0, 45.0, 60.0)
    # -- WOOF options (design E3).  Each is off by default, so a default
    # config is NOAA's product and the oracle comparison applies to it; a
    # config with any of them on leaves NOAA's product and says so in its
    # receipt (:func:`config_record`).
    #: Lower the heating's bottom from ``max(krad_bot, PBL top)`` to the PBL
    #: top itself in columns where at least ``pbl_extension_depth_hpa`` of
    #: depth is observed (echo or observed no echo, after the cone fill).
    #: NOAA's floor of model level 7 keeps a shallow boundary layer's top
    #: unheated; the extension reaches down to it where the radar saw enough
    #: of the column to say what is there.
    pbl_extension: bool = False
    pbl_extension_depth_hpa: float = 200.0
    #: Force the tendency back to exactly zero at every covered point the
    #: radar observed as no echo, AFTER NOAA's two smoothing passes.  NOAA
    #: smooths the tendency over observed-clear points too, so heat bleeds
    #: one to two cells into air the radar saw as empty
    #: (:func:`suppression_receipt` counts it); with this on, "the radar saw
    #: nothing" always means "no latent heating here".
    strict_suppression: bool = False

    def is_noaa(self) -> bool:
        """True when every value is NOAA's (the oracle applies)."""
        return self == RadarTtenConfig()


# ``radar_ref2tten.f90:113-127``, evaluated in the order the PARAMETER
# statements state, in binary64 as the compiler folds them.
_R_P = 8.31451
_MD_P = 0.0289645
_RD_P = _R_P / _MD_P
_CPD_P = 3.5 * _RD_P
_CPOVR_P = _CPD_P / _RD_P
_LV_P = 2.501e6
_LF0_P = 0.3335e6
#: ``constants.f90``: ``rd = 287.04`` (``:325, :333``, the regional branch
#: the driver selects at ``gsdcloudanalysis_ref2tten.f90:155``) over
#: ``cp = 1004.6`` (``:92``), formed at ``:377``.
_RD_OVER_CP = 287.04 / 1004.6

# ``build_missing_REFcone.f90:68-70`` (km).  The DATA constants are default
# reals stored into a double array, so each is the float32 value widened.
_CONE_LEVELS_KM = (
    0.2, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
    3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5,
    9, 10, 11, 12, 13, 14, 15, 16)

# ``build_missing_REFcone.f90:114-149``: the summer profiles, one row per
# class of column-maximum reflectivity (20-25, 25-30, ..., 45-50 dBZ).
_CONE_SUMMER = (
    (0.883, 0.870, 0.879, 0.892, 0.904, 0.912, 0.913, 0.915, 0.924, 0.936,
     0.946, 0.959, 0.984, 0.999, 1.000, 0.995, 0.988, 0.978, 0.962, 0.940,
     0.916, 0.893, 0.865, 0.839, 0.778, 0.708, 0.666, 0.686, 0.712, 0.771,
     0.833),
    (0.836, 0.874, 0.898, 0.915, 0.927, 0.938, 0.945, 0.951, 0.960, 0.970,
     0.980, 0.989, 1.000, 0.995, 0.968, 0.933, 0.901, 0.861, 0.822, 0.783,
     0.745, 0.717, 0.683, 0.661, 0.614, 0.564, 0.538, 0.543, 0.578, 0.633,
     0.687),
    (0.870, 0.885, 0.914, 0.931, 0.943, 0.954, 0.967, 0.975, 0.982, 0.989,
     0.995, 1.000, 0.998, 0.973, 0.918, 0.850, 0.791, 0.735, 0.690, 0.657,
     0.625, 0.596, 0.569, 0.544, 0.510, 0.479, 0.461, 0.460, 0.477, 0.522,
     0.570),
    (0.871, 0.895, 0.924, 0.948, 0.961, 0.971, 0.978, 0.983, 0.988, 0.992,
     0.997, 1.000, 0.995, 0.966, 0.913, 0.848, 0.781, 0.719, 0.660, 0.611,
     0.576, 0.542, 0.523, 0.513, 0.481, 0.448, 0.416, 0.402, 0.417, 0.448,
     0.491),
    (0.875, 0.895, 0.914, 0.936, 0.942, 0.951, 0.964, 0.979, 0.990, 0.998,
     1.000, 0.992, 0.961, 0.905, 0.834, 0.772, 0.722, 0.666, 0.618, 0.579,
     0.545, 0.518, 0.509, 0.483, 0.419, 0.398, 0.392, 0.403, 0.423, 0.480,
     0.440),
    (0.926, 0.920, 0.948, 0.975, 0.988, 0.989, 0.995, 0.997, 1.000, 1.000,
     0.997, 0.991, 0.970, 0.939, 0.887, 0.833, 0.788, 0.741, 0.694, 0.655,
     0.611, 0.571, 0.551, 0.537, 0.507, 0.470, 0.432, 0.410, 0.420, 0.405,
     0.410),
)
_CONE_LEVELS = 31
_CONE_CLASSES = 6


def _cone_tables():
    """``(levels_m, profiles)`` as NOAA's program holds them: float64 arrays
    whose values are float32 constants widened, the levels then multiplied
    by 1000 in float64 (``build_missing_REFcone.f90:164-166``)."""
    levels = np.asarray(_CONE_LEVELS_KM, dtype=np.float32).astype(np.float64)
    levels = levels * 1000.0
    profiles = np.asarray(_CONE_SUMMER, dtype=np.float32).astype(np.float64)
    if levels.shape != (_CONE_LEVELS,) or profiles.shape != (
            _CONE_CLASSES, _CONE_LEVELS):
        raise AssertionError("cone-fill tables lost a value")
    return levels, np.ascontiguousarray(profiles)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
#
# Index convention: arrays are C-order (nz, ny, nx), which is the same
# memory as NOAA's Fortran (nlon, nlat, nsig).  Comments name Fortran's
# one-based indices where a loop bound is being matched.
#
# Every division is __fdiv_rn or __ddiv_rn and every other arithmetic node
# is its own __f*_rn or __d*_rn call, so the compiler can neither turn a
# division by a constant into a reciprocal multiply nor fuse a multiply
# into an add.

_SOURCE = r"""
__device__ __forceinline__ float rtt_thetav(const float q, const float t) {
    // pbl_height.f90:74-75
    const float qsp = __fdiv_rn(q, __fadd_rn(1.0f, q));
    return __fmul_rn(t, __fadd_rn(1.0f, __fmul_rn(0.61f, qsp)));
}

__device__ __forceinline__ float rtt_temperature(
        const float theta, const float p_hpa, const double rd_over_cp) {
    // radar_ref2tten.f90:151, :181 -- a float32 result of float64 work.
    return (float)__dmul_rn(
        (double)theta, glibc_pow(__ddiv_rn((double)p_hpa, 1000.0), rd_over_cp));
}

__device__ __forceinline__ int rtt_krad_bot(const float krad_bot_in, const float pblh) {
    // radar_ref2tten.f90:180, :283
    return __float2int_rz(__fadd_rn(fmaxf(krad_bot_in, pblh), 0.5f));
}

__device__ __forceinline__ float rtt_bottom_in(
        const float krad_bot_in, const signed char* __restrict__ ext, const int col) {
    // WOOF PBL extension: a column flagged by rtt_observed_depth takes the
    // heating's bottom from its PBL top alone (level 1 never wins against a
    // PBL top, which rtt_pbl_height keeps at 2 or above).  Unflagged
    // columns keep NOAA's floor, so an all-zero flag array is NOAA's code.
    return ext[col] ? 1.0f : krad_bot_in;
}

extern "C" __global__ void rtt_observed_depth(
        const float* __restrict__ ref, const float* __restrict__ p,
        signed char* __restrict__ ext, const double depth_hpa,
        const int nz, const int ny, const int nx) {
    // WOOF PBL extension: hPa of the column the radar observed (echo or
    // observed no echo, i.e. ref > -100, after the cone fill), summed with
    // NOAA's own layer depth 0.5 * (p(k-1) - p(k+1)) over levels 2..nz-1.
    const int ncol = ny * nx;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    double dpint = 0.0;
    for (int kf = 2; kf <= nz - 1; ++kf) {
        const long long at = (long long)(kf - 1) * ncol + col;
        if (ref[at] > -100.0f)
            dpint = __dadd_rn(dpint, __dmul_rn(
                0.5, (double)__fsub_rn(p[at - ncol], p[at + ncol])));
    }
    ext[col] = (dpint >= depth_hpa) ? 1 : 0;
}

extern "C" __global__ void rtt_background(
        const float* __restrict__ thb, const float* __restrict__ thp,
        const float* __restrict__ p, const float* __restrict__ phb,
        const float* __restrict__ php,
        float* __restrict__ theta, float* __restrict__ p_hpa,
        float* __restrict__ h_agl,
        const float g, const int thb_full, const int phb_full,
        const int nz, const int ncol) {
    const long long n = (long long)nz * ncol;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int k = (int)(idx / ncol);
    const long long col = idx - (long long)k * ncol;
    theta[idx] = __fadd_rn(thb[thb_full ? idx : k], thp[idx]);
    p_hpa[idx] = __fdiv_rn(p[idx], 100.0f);
    const float z0 = __fdiv_rn(__fadd_rn(phb[phb_full ? col : 0], php[col]), g);
    const float zlo = __fdiv_rn(
        __fadd_rn(phb[phb_full ? idx : k], php[idx]), g);
    const float zhi = __fdiv_rn(
        __fadd_rn(phb[phb_full ? idx + ncol : k + 1], php[idx + ncol]), g);
    h_agl[idx] = __fsub_rn(__fmul_rn(0.5f, __fadd_rn(zlo, zhi)), z0);
}

extern "C" __global__ void rtt_adapt(
        const float* __restrict__ z_obs, const signed char* __restrict__ z_mask,
        const signed char* __restrict__ z0_mask, float* __restrict__ ref,
        const int has_clear, const long long n) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float out = -99999.0f;
    if (z_mask[idx] != 0) out = z_obs[idx];
    else if (has_clear && z0_mask[idx] != 0) out = -99.0f;
    ref[idx] = out;
}

extern "C" __global__ void rtt_pbl_height(
        const float* __restrict__ q, const float* __restrict__ t,
        float* __restrict__ pblh, const int nz, const int ncol) {
    // pbl_height.f90:70-91, one column per thread, float32 throughout.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const float thsfc = rtt_thetav(q[col], t[col]);
    const float lim = __fadd_rn(thsfc, 1.0f);
    float out = 0.0f;
    int k = 1;                                   // Fortran level
    while (fabsf(out) < 0.0001f && k < nz - 2) {
        const long long at = (long long)(k - 1) * ncol + col;
        const float thk = rtt_thetav(q[at], t[at]);
        if (thk > lim) {
            // Level 1 never passes: thetav(1) > thetav(1) + 1 is false.
            const float thm = rtt_thetav(q[at - ncol], t[at - ncol]);
            out = __fsub_rn((float)k, __fdiv_rn(
                __fsub_rn(thk, lim), fmaxf(__fsub_rn(thk, thm), 0.01f)));
        }
        k += 1;
    }
    if (fabsf(out) < 0.0001f) out = 2.0f;
    pblh[col] = out;
}

__device__ __forceinline__ double rtt_cone_profile(
        const float height, const double column_max, const int mref,
        const double* __restrict__ levels, const double* __restrict__ profiles) {
    // build_missing_REFcone.f90:203-222 for one level; -9999.9 outside the
    // summer window [newlvlAll(3), 12000 m).
    const double hg = (double)height;
    if (!(hg >= levels[2] && hg < 12000.0)) return -9999.9;
    int ilvl = 0;
    for (int k = 0; k < 30; ++k)
        if (hg >= levels[k] && hg < levels[k + 1]) ilvl = k;
    const double* row = profiles + (mref - 1) * 31;
    const double upref = __dmul_rn(row[ilvl + 1], column_max);
    const double downref = __dmul_rn(row[ilvl], column_max);
    const double wght = __ddiv_rn(
        __dsub_rn(hg, levels[ilvl]), __dsub_rn(levels[ilvl + 1], levels[ilvl]));
    return __dadd_rn(__dmul_rn(__dsub_rn(1.0, wght), downref),
                     __dmul_rn(wght, upref));
}

extern "C" __global__ void rtt_cone_fill(
        float* __restrict__ ref, const float* __restrict__ h,
        const float* __restrict__ pblh,
        const double* __restrict__ levels, const double* __restrict__ profiles,
        const int nz, const int ny, const int nx) {
    // build_missing_REFcone.f90:168-243, interior columns only.
    const int ncol = ny * nx;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const int j = col / nx, i = col - j * nx;
    if (i < 1 || i > nx - 2 || j < 1 || j > ny - 2) return;
#define RTT_REF(kf) ref[(long long)((kf) - 1) * ncol + col]
#define RTT_HGT(kf) h[(long long)((kf) - 1) * ncol + col]
    // :176.  A boundary-layer top from rtt_pbl_height lies in [1, nz - 2);
    // the two clamps only keep a caller's own array from walking the loops
    // outside the column.
    const int krad_bot = __float2int_rz(__fadd_rn(pblh[col], 0.5f));
    const int klo = krad_bot < 1 ? 1 : krad_bot;
    const int khi = krad_bot > nz ? nz : krad_bot;
    int ifmissing = 0;
    double maxref = -9999.0;
    for (int k2 = nz / 2; k2 >= klo; --k2) {                    // :180-184
        if ((double)RTT_REF(k2 + 1) >= 20.0 && (double)RTT_REF(k2) < -100.0)
            ifmissing = k2;
        if ((double)RTT_REF(k2) >= maxref) maxref = (double)RTT_REF(k2);
    }
    if (ifmissing > 1) {                                        // :185-188
        for (int k2 = khi; k2 >= 1; --k2)
            if ((double)RTT_REF(k2) > maxref) maxref = (double)RTT_REF(k2);
    }
    if (ifmissing > 1 && maxref > 19.0) {                       // :194
        int mref = __double2int_rz(__ddiv_rn(__dsub_rn(maxref, 20.0), 5.0)) + 1;
        if (mref > 6) mref = 6;
        if (mref < 1) mref = 1;
        const float above = RTT_REF(ifmissing + 1);
        const double diff = __dsub_rn((double)above, rtt_cone_profile(
            RTT_HGT(ifmissing + 1), maxref, mref, levels, profiles));  // :226
        if (fabs(diff) < 10.0) {                                // :227-230
            for (int k2 = ifmissing; k2 >= klo; --k2)
                RTT_REF(k2) = (float)__dadd_rn(rtt_cone_profile(
                    RTT_HGT(k2), maxref, mref, levels, profiles), diff);
        } else {                                                // :232-234
            for (int k2 = ifmissing; k2 >= klo; --k2) RTT_REF(k2) = above;
        }
    }
    for (int k2 = 1; k2 <= khi; ++k2) RTT_REF(k2) = -99999.0f;  // :239-241
#undef RTT_REF
#undef RTT_HGT
}

extern "C" __global__ void rtt_probable_convection(
        const float* __restrict__ ref, const float* __restrict__ p,
        const float* __restrict__ t, signed char* __restrict__ probable,
        const double threshold, const double warm_k, const double cold_depth,
        const double rd_over_cp, const int nz, const int ny, const int nx) {
    // radar_ref2tten.f90:146-161.
    const int ncol = ny * nx;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const int j = col / nx, i = col - j * nx;
    signed char hit = 0;
    if (i >= 1 && i <= nx - 2 && j >= 1 && j <= ny - 2) {
        double dpint = 0.0;
        for (int kf = 2; kf <= nz - 1; ++kf) {
            const long long at = (long long)(kf - 1) * ncol + col;
            const float tbk = rtt_temperature(t[at], p[at], rd_over_cp);
            const double r = (double)ref[at];
            if ((double)tbk >= warm_k && r >= threshold) hit = 1;
            if ((double)tbk < warm_k && r >= threshold)
                dpint = __dadd_rn(dpint, __dmul_rn(
                    0.5, (double)__fsub_rn(p[at - ncol], p[at + ncol])));
        }
        if (dpint >= cold_depth) hit = 1;
    }
    probable[col] = hit;
}

extern "C" __global__ void rtt_tendency(
        const float* __restrict__ ref, const float* __restrict__ p,
        const float* __restrict__ t, const float* __restrict__ pblh,
        const signed char* __restrict__ ext, double* __restrict__ tten,
        const float krad_bot_in, const double echo_floor, const double warm_k,
        const double warm_min_dbz, const double z_scale, const double z_divisor,
        const double z_factor, const double inv_cpovr, const double latent,
        const double period_min, const double cpd, const double cap,
        const double rd_over_cp, const int nz, const int ny, const int nx) {
    // radar_ref2tten.f90:164-217; zero outside the loop bounds.
    const int ncol = ny * nx;
    const long long n = (long long)nz * ncol;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int k = (int)(idx / ncol);
    const int col = (int)(idx - (long long)k * ncol);
    const int j = col / nx, i = col - j * nx;
    double out = 0.0;
    if (k >= 1 && k <= nz - 2 && i >= 1 && i <= nx - 2 && j >= 1 && j <= ny - 2) {
        const float r = ref[idx];
        const double rd = (double)r;
        if (rd < echo_floor && r > -100.0f) {                   // :182
            out = 0.0;
        } else if (rd >= echo_floor) {                          // :184
            const float tbk = rtt_temperature(t[idx], p[idx], rd_over_cp);
            const bool skip = ((double)tbk > warm_k) && (rd < warm_min_dbz);
            if (!skip && (k + 1) >= rtt_krad_bot(
                    rtt_bottom_in(krad_bot_in, ext, col), pblh[col])) {
                // :203
                const double addsnow = __dmul_rn(__ddiv_rn(
                    glibc_pow(10.0, __ddiv_rn(rd, z_scale)), z_divisor), z_factor);
                // :205-207
                const double tt = __dmul_rn(
                    glibc_pow(__ddiv_rn(1000.0, (double)p[idx]), inv_cpovr),
                    __ddiv_rn(__dmul_rn(latent, addsnow),
                              __dmul_rn(__dmul_rn(period_min, 60.0), cpd)));
                out = fmin(cap, fmax(-cap, tt));                // :210
            }
        }
    }
    tten[idx] = out;
}

extern "C" __global__ void rtt_smooth_interior(
        const double* __restrict__ src, double* __restrict__ dst,
        const double s1, const double s2, const double s3,
        const int nz, const int ny, const int nx) {
    // smooth.f90:60-81.  The two-row hold buffer means every interior value
    // is built from the UNSMOOTHED field, so the interior is one parallel
    // stencil.  Edge and corner points are copied; rtt_smooth_edges then
    // runs the edge recurrences on the copy.
    const int ncol = ny * nx;
    const long long n = (long long)nz * ncol;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int k = (int)(idx / ncol);
    const int col = (int)(idx - (long long)k * ncol);
    const int j = col / nx, i = col - j * nx;
    if (i < 1 || i > nx - 2 || j < 1 || j > ny - 2) {
        dst[idx] = src[idx];
        return;
    }
    const double* f = src + idx;
    const double sum1 = __dadd_rn(__dadd_rn(__dadd_rn(
        f[nx - 1], f[-nx - 1]), f[nx + 1]), f[-nx + 1]);        // :65-66
    const double sum2 = __dadd_rn(__dadd_rn(__dadd_rn(
        f[nx], f[1]), f[-nx]), f[-1]);                          // :67-68
    dst[idx] = __dadd_rn(__dadd_rn(__dmul_rn(s1, sum1), __dmul_rn(s2, sum2)),
                         __dmul_rn(s3, f[0]));                  // :69
}

extern "C" __global__ void rtt_smooth_edges(
        double* __restrict__ field, const double s4, const double s5,
        const int nz, const int ny, const int nx) {
    // smooth.f90:83-95.  Each edge is updated in place in index order, so
    // a point reads its lower neighbour already smoothed and its upper
    // neighbour not yet: a recurrence, run here as one thread's loop.  The
    // four edges of a level share only the corners, which never change.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nz * 4) return;
    const int k = t / 4, edge = t - k * 4;
    double* g = field + (long long)k * ny * nx;
    if (edge < 2) {
        double* row = g + (edge == 0 ? 0 : (long long)(ny - 1) * nx);
        for (int i = 1; i <= nx - 2; ++i)
            row[i] = __dadd_rn(__dmul_rn(s4, row[i]),
                               __dmul_rn(s5, __dadd_rn(row[i - 1], row[i + 1])));
    } else {
        double* colp = g + (edge == 2 ? 0 : nx - 1);
        for (int j = 1; j <= ny - 2; ++j) {
            const long long at = (long long)j * nx;
            colp[at] = __dadd_rn(__dmul_rn(s4, colp[at]),
                                 __dmul_rn(s5, __dadd_rn(colp[at - nx], colp[at + nx])));
        }
    }
}

extern "C" __global__ void rtt_store(
        const double* __restrict__ tten, const float* __restrict__ ref,
        float* __restrict__ ges, const long long n) {
    // radar_ref2tten.f90:227-234.
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float out = (float)tten[idx];
    if ((double)ref[idx] <= -200.0) out = -99999.0f;
    ges[idx] = out;
}

extern "C" __global__ void rtt_strict_clear(
        float* __restrict__ ges, const float* __restrict__ ref,
        const double echo_floor, const long long n_below) {
    // WOOF strict suppression, after NOAA's whole sequence: every level
    // below the top that the radar observed as no echo holds exactly zero
    // tendency, whether NOAA's smoothing bled heat into it or a stratiform
    // column handed it back to the model.
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_below) return;
    const float r = ref[idx];
    if (r > -100.0f && (double)r < echo_floor) ges[idx] = 0.0f;
}

extern "C" __global__ void rtt_column_flag(
        const double* __restrict__ tten, const float* __restrict__ ref,
        const float* __restrict__ p, const float* __restrict__ pblh,
        const signed char* __restrict__ ext,
        const signed char* __restrict__ probable, float* __restrict__ ges,
        const float krad_bot_in, const double coverage_depth,
        const double nearby, const int convection_only,
        const int nz, const int ny, const int nx) {
    // radar_ref2tten.f90:260-346, every column.
    const int ncol = ny * nx;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const int j = col / nx, i = col - j * nx;
    double radmax = 0.0, dpint = 0.0;
    for (int kf = 2; kf <= nz - 1; ++kf) {                      // :271-278
        const long long at = (long long)(kf - 1) * ncol + col;
        double v = tten[at];
        if ((double)ref[at] <= -200.0) v = -99999.0;
        if (v > -15.0) {
            dpint = __dadd_rn(dpint, __dmul_rn(
                0.5, (double)__fsub_rn(p[at - ncol], p[at + ncol])));
            radmax = fmax(radmax, v);
        }
    }
    double radyn = -10.0;
    if (dpint >= coverage_depth) {                              // :279-288
        radyn = 0.0;
        if (radmax > nearby) radyn = 1.0;
        if (fabs(radyn) < 0.00001) {
            int kb = rtt_krad_bot(rtt_bottom_in(krad_bot_in, ext, col),
                                  pblh[col]);
            if (kb < 1) kb = 1;
            for (int kf = kb; kf <= nz - 1; ++kf)
                ges[(long long)(kf - 1) * ncol + col] = 0.0f;
        }
    }
    const long long top = (long long)(nz - 1) * ncol + col;
    ges[top] = (float)radyn;                                    // :305-309
    if (convection_only && i >= 1 && i <= nx - 2 && j >= 1 && j <= ny - 2
            && radyn > 0.9 && probable[col] == 0) {             // :315-326
        ges[top] = -10.0f;
        for (int kf = 2; kf <= nz - 1; ++kf)
            ges[(long long)(kf - 1) * ncol + col] = -99999.0f;
    }
    for (int kf = 1; kf <= nz; ++kf) {                          // :340-346
        const long long at = (long long)(kf - 1) * ncol + col;
        if ((double)ges[at] <= -200.0) ges[at] = -20.0f;
    }
}

extern "C" __global__ void rtt_vinterp(
        const float* __restrict__ mosaic, const float* __restrict__ h_agl,
        const float* __restrict__ terrain, const double* __restrict__ levels,
        float* __restrict__ ref, const int nlevels,
        const int nz, const int ny, const int nx) {
    // vinterp_radar_ref.f90:84, :106-132, interior columns.  Edge columns
    // keep the -99999 fill until rtt_vinterp_edges copies them in.
    const int ncol = ny * nx;
    const long long n = (long long)nz * ncol;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int k = (int)(idx / ncol);
    const int col = (int)(idx - (long long)k * ncol);
    const int j = col / nx, i = col - j * nx;
    if (i < 1 || i > nx - 2 || j < 1 || j > ny - 2) {
        ref[idx] = -99999.0f;
        return;
    }
    const double hg = (double)__fadd_rn(h_agl[idx], terrain[col]);   // :109
    double out = -99999.0;
    if (hg >= levels[0] && hg < levels[nlevels - 1]) {               // :110
        int ilvl = 0;
        for (int m = 0; m < nlevels - 1; ++m)
            if (hg >= levels[m] && hg < levels[m + 1]) ilvl = m;
        const double up = (double)mosaic[(long long)(ilvl + 1) * ncol + col];
        const double down = (double)mosaic[(long long)ilvl * ncol + col];
        double value;
        if (fabs(up) < 90.0 && fabs(down) < 90.0) {                  // :116-119
            const double wght = __ddiv_rn(__dsub_rn(hg, levels[ilvl]),
                                          __dsub_rn(levels[ilvl + 1], levels[ilvl]));
            value = __dadd_rn(__dmul_rn(__dsub_rn(1.0, wght), down),
                              __dmul_rn(wght, up));
        } else if (fabs(__dadd_rn(up, 99.0)) < 0.1
                   || fabs(__dadd_rn(down, 99.0)) < 0.1) {           // :120-122
            value = -99.0;
        } else {
            value = -99999.0;                                        // :124
        }
        out = fmax(-99999.0, value);                                 // :126
    }
    ref[idx] = (float)out;
}

extern "C" __global__ void rtt_vinterp_edges(
        float* __restrict__ ref, const int nz, const int ny, const int nx) {
    // vinterp_radar_ref.f90:135-148.  Every value read is an interior one,
    // so the copies are independent.  :147 is kept as NOAA wrote it: after
    // the DO j loop j is nlat, so it sets (1, nlat) from (2, 2) and leaves
    // (1, 1) at the -99999 fill.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int per_level = 2 * (nx - 2) + 2 * (ny - 2) + 3;
    if (t >= nz * per_level) return;
    const int k = t / per_level;
    int e = t - k * per_level;
    float* g = ref + (long long)k * ny * nx;
#define RTT_AT(jj, ii) g[(long long)(jj) * nx + (ii)]
    if (e < nx - 2) { const int i = e + 1; RTT_AT(0, i) = RTT_AT(1, i); return; }
    e -= nx - 2;
    if (e < nx - 2) { const int i = e + 1; RTT_AT(ny - 1, i) = RTT_AT(ny - 2, i); return; }
    e -= nx - 2;
    if (e < ny - 2) { const int j = e + 1; RTT_AT(j, 0) = RTT_AT(j, 1); return; }
    e -= ny - 2;
    if (e < ny - 2) { const int j = e + 1; RTT_AT(j, nx - 1) = RTT_AT(j, nx - 2); return; }
    e -= ny - 2;
    if (e == 0) RTT_AT(ny - 1, nx - 1) = RTT_AT(ny - 2, nx - 2);     // :144
    else if (e == 1) RTT_AT(0, nx - 1) = RTT_AT(1, nx - 2);          // :145
    else RTT_AT(ny - 1, 0) = RTT_AT(1, 1);                           // :146-147
#undef RTT_AT
}

extern "C" __global__ void rtt_apply(
        float* __restrict__ thp, const float* __restrict__ thp_before,
        const float* __restrict__ slot, const float dt, const int ring,
        const int nz, const int ny, const int nx,
        const int j0, const int i0, const int gny, const int gnx) {
    // module_big_step_utilities_em.F:5991-6005.  `thp` already holds the
    // microphysics result, which is what an uncovered point keeps (:6003);
    // a covered point below the top level takes the tendency instead
    // (:5996, :6000).  The top level holds the column flag, never a
    // tendency, and the specified-zone ring is outside the tile WRF runs
    // this routine on.  (j0, i0) is this array's origin in the domain and
    // (gny, gnx) the domain's extent: a card's slab tests the ring in the
    // DOMAIN's indices, so a seam column is heated exactly as the same
    // column of a one-card run.  A whole domain passes (0, 0, ny, nx).
    const int ncol = ny * nx;
    const long long n = (long long)nz * ncol;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int k = (int)(idx / ncol);
    const int col = (int)(idx - (long long)k * ncol);
    const int j = col / nx + j0, i = col % nx + i0;
    if (k >= nz - 1 || i < ring || i >= gnx - ring || j < ring || j >= gny - ring)
        return;
    const float v = slot[idx];
    if (v >= -1.0f && v <= 1.0f)
        thp[idx] = __fadd_rn(thp_before[idx], __fmul_rn(v, dt));
}

// ---------------------------------------------------------------------------
// Latent heat nudging (the comparison arm, design E3; ICON/COSMO style).
// Not NOAA code and not graded against an oracle: ordinary float arithmetic.
// ---------------------------------------------------------------------------

extern "C" __global__ void lhn_observed_rate(
        const float* __restrict__ ref, const float* __restrict__ h_agl,
        float* __restrict__ rate, const float max_height_m,
        const float zr_a, const float zr_b, const int nz, const int ncol) {
    // Surface rain rate (mm/h) per column from the LOWEST observed level at
    // or below max_height_m: Z = a R^b on echo, 0 on observed no echo, -1
    // where no level that low is observed (no information).
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    float out = -1.0f;
    for (int k = 0; k < nz; ++k) {
        const long long at = (long long)k * ncol + col;
        if (h_agl[at] > max_height_m) break;
        const float r = ref[at];
        if (r > -100.0f) {
            out = (r >= 0.001f)
                ? powf(__fdiv_rn(powf(10.0f, __fmul_rn(r, 0.1f)), zr_a),
                       __fdiv_rn(1.0f, zr_b)) : 0.0f;
            break;
        }
    }
    rate[col] = out;
}

__device__ __forceinline__ float lhn_qsat(const float t, const float p) {
    // Bolton (1980) over water, Pa and K.
    const float es = 611.2f * expf(17.67f * (t - 273.15f) / (t - 29.65f));
    return 0.622f * es / fmaxf(p - es, 1.0f);
}

extern "C" __global__ void lhn_apply(
        float* __restrict__ thp, const float* __restrict__ thp_before,
        float* __restrict__ qv, const float* __restrict__ thb,
        const float* __restrict__ p, const float* __restrict__ rate_obs,
        const float* __restrict__ rainncv, const float dt,
        const float alpha_min, const float alpha_max, const float threshold,
        const float cap_k_per_s, const int thb_full, const int ring,
        unsigned long long* __restrict__ counts,
        const int nz, const int ny, const int nx) {
    // The model's own microphysics heating at this step, scaled by the
    // ratio of observed to model surface rain rate, in heating points
    // only, capped, with relative humidity kept (vapour rescaled to the
    // new saturation value).  counts: [up, down].
    const int ncol = ny * nx;
    const long long n = (long long)nz * ncol;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int k = (int)(idx / ncol);
    const int col = (int)(idx - (long long)k * ncol);
    const int j = col / nx, i = col - j * nx;
    if (k >= nz - 1 || i < ring || i >= nx - ring || j < ring || j >= ny - ring)
        return;
    const float ro = rate_obs[col];
    if (ro < 0.0f) return;                              // no information
    const float rm = rainncv[col] * 3600.0f / dt;       // mm per step -> mm/h
    float alpha;
    if (ro >= threshold && rm >= threshold) {
        alpha = fminf(fmaxf(ro / rm, alpha_min), alpha_max);
    } else if (ro < threshold && rm >= threshold) {
        alpha = alpha_min;                              // model rains, radar does not
    } else {
        return;   // model dry: no heating profile of its own to scale
    }
    const float lh = thp[idx] - thp_before[idx];
    if (!(lh > 0.0f)) return;                           // in-cloud heating only
    const float cap = cap_k_per_s * dt;
    const float inc = fminf(fmaxf((alpha - 1.0f) * lh, -cap), cap);
    if (inc == 0.0f) return;
    const float base = thb[thb_full ? idx : k];
    const float exner = powf(__fmul_rn(p[idx], 1.0e-5f), 0.285714f);
    const float t_old = (base + thp[idx]) * exner;
    const float t_new = (base + thp[idx] + inc) * exner;
    qv[idx] = qv[idx] * (lhn_qsat(t_new, p[idx]) / lhn_qsat(t_old, p[idx]));
    thp[idx] = thp[idx] + inc;
    atomicAdd(&counts[inc > 0.0f ? 0 : 1], 1ULL);
}
"""

#: Threads per block.  Every kernel is one thread per output element (or
#: per column, or per edge), so the launch shape cannot move a result.
_THREADS = 256


def _full_source() -> str:
    """The shared binary64 libm transcription, then this module's kernels."""
    from woof.core import kernels as kernel_loader

    header = Path(kernel_loader.__file__).parent / "glibc_flt64.cuh"
    return header.read_text(encoding="utf-8") + "\n" + _SOURCE


def _require_device():
    """CuPy with a usable CUDA device, or a refusal that says why.

    There is no host implementation to fall back to.  The breakage a silent
    fallback would cause: two implementations of one product, of which only
    one is graded against NOAA's code.
    """
    from woof.local_gpu import no_local_gpu

    if no_local_gpu():
        raise RadarTtenError(
            "radar latent heating runs only as CUDA kernels and "
            "GPUWM_NO_LOCAL_GPU forbids this process the local device. "
            "There is no host implementation to fall back to")
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RadarTtenError(
            "radar latent heating runs only as CUDA kernels and this "
            f"process has no usable CUDA device ({type(exc).__name__}: "
            f"{exc}). There is no host implementation to fall back to"
        ) from exc
    if count < 1:
        raise RadarTtenError(
            "radar latent heating runs only as CUDA kernels and the CUDA "
            "runtime reports zero devices. There is no host implementation "
            "to fall back to")
    return cp


@lru_cache(maxsize=None)
def _kernel(name: str):
    import cupy as cp

    return cp.RawKernel(_full_source(), name, options=("-std=c++17",))


def _launch(name: str, count: int, args: tuple) -> None:
    blocks = (int(count) + _THREADS - 1) // _THREADS
    _kernel(name)((max(blocks, 1),), (_THREADS,), args)


def _volume(cp, name: str, value, shape=None):
    """A float32 C-contiguous device volume, or a refusal naming the field."""
    if not isinstance(value, cp.ndarray):
        raise RadarTtenError(
            f"{name} must be a device array; got {type(value).__name__}. "
            "The builder has no host path")
    if value.dtype != np.float32 or value.ndim != 3:
        raise RadarTtenError(
            f"{name} must be float32 (nz, ny, nx); got {value.dtype} "
            f"{value.shape}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise RadarTtenError(
            f"{name} has shape {tuple(value.shape)}, expected {tuple(shape)}")
    return cp.ascontiguousarray(value)


def _require_extent(shape) -> None:
    nz, ny, nx = (int(v) for v in shape)
    if nz < 4 or ny < 3 or nx < 3:
        raise RadarTtenError(
            f"grid {(nz, ny, nx)} is smaller than NOAA's loops assume "
            "(at least 4 levels and 3 by 3 columns): the interior loops "
            "run 2..n-1 and the boundary-layer search stops at nz-2, so a "
            "smaller grid would produce a field with no interior")


# --------------------------------------------------------------------------
# the builder's steps
# --------------------------------------------------------------------------

def pbl_height(qv, theta):
    """NOAA's ``calc_pbl_height``: the boundary-layer top per column, as a
    fractional (one-based) level index, float32 ``(ny, nx)``."""
    cp = _require_device()
    theta = _volume(cp, "theta", theta)
    qv = _volume(cp, "qv", qv, theta.shape)
    _require_extent(theta.shape)
    nz, ny, nx = theta.shape
    out = cp.empty((ny, nx), dtype=np.float32)
    _launch("rtt_pbl_height", ny * nx,
            (qv, theta, out, np.int32(nz), np.int32(ny * nx)))
    return out


def cone_fill(ref, height_agl_m, pblh):
    """NOAA's ``build_missing_REFcone`` on a COPY of ``ref``.

    Fills missing reflectivity under the radar cone from the summer profile
    and sets every level up to the boundary-layer top to no coverage, on
    interior columns.  Returns the new reflectivity volume.
    """
    cp = _require_device()
    ref = _volume(cp, "ref", ref)
    height_agl_m = _volume(cp, "height_agl_m", height_agl_m, ref.shape)
    _require_extent(ref.shape)
    nz, ny, nx = ref.shape
    if not isinstance(pblh, cp.ndarray) or pblh.shape != (ny, nx) \
            or pblh.dtype != np.float32:
        raise RadarTtenError(f"pblh must be a float32 device {(ny, nx)}")
    levels, profiles = _cone_tables()
    out = ref.copy()
    _launch("rtt_cone_fill", ny * nx,
            (out, height_agl_m, cp.ascontiguousarray(pblh),
             cp.asarray(levels), cp.asarray(profiles),
             np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


#: Mosaic level sets ``vinterp_radar_ref.f90`` accepts, in km above mean
#: sea level (``:63-70``), keyed by level count.
MOSAIC_LEVELS_KM = {
    21: (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
         15, 16, 17),
    31: (0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5,
         5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15, 16, 18),
    33: (0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5,
         5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19),
}


def vinterp_mosaic(mosaic, height_agl_m, terrain_m):
    """NOAA's ``vinterp_radar_ref``: a reflectivity mosaic on its fixed
    heights above sea level to the model's levels.

    ``mosaic`` is a float32 device volume ``(levels, ny, nx)`` holding one
    of the level sets of :data:`MOSAIC_LEVELS_KM` (dBZ, ``-99`` no echo,
    anything at or beyond 90 in magnitude other than ``-99`` missing);
    ``height_agl_m`` is ``(nz, ny, nx)`` and ``terrain_m`` ``(ny, nx)``.
    Returns reflectivity on the model grid in the builder's convention.  A
    point between two mosaic levels takes the linear value when both are
    valid, no echo when either is ``-99``, and no coverage otherwise; a
    point below the first level or at or above the last is no coverage.
    Edge columns are copied from their inner neighbours as NOAA copies
    them, including its corner quirk (``:147``): column (1, 1) keeps no
    coverage and column (1, ny) takes column (2, 2).
    """
    cp = _require_device()
    height_agl_m = _volume(cp, "height_agl_m", height_agl_m)
    _require_extent(height_agl_m.shape)
    nz, ny, nx = (int(v) for v in height_agl_m.shape)
    if not isinstance(mosaic, cp.ndarray) or mosaic.dtype != np.float32 \
            or mosaic.ndim != 3 or tuple(mosaic.shape[1:]) != (ny, nx):
        raise RadarTtenError(
            f"mosaic must be a float32 device (levels, {ny}, {nx}) volume")
    nlevels = int(mosaic.shape[0])
    if nlevels not in MOSAIC_LEVELS_KM:
        raise RadarTtenError(
            f"a {nlevels}-level mosaic has no level table: NOAA's routine "
            f"knows {sorted(MOSAIC_LEVELS_KM)} and stops on any other "
            "count, because interpolating against the wrong heights puts "
            "every echo at the wrong altitude")
    if not isinstance(terrain_m, cp.ndarray) or terrain_m.dtype != np.float32 \
            or tuple(terrain_m.shape) != (ny, nx):
        raise RadarTtenError(f"terrain_m must be a float32 device {(ny, nx)}")
    levels = (np.asarray(MOSAIC_LEVELS_KM[nlevels], dtype=np.float32)
              .astype(np.float64) * 1000.0)
    out = cp.empty((nz, ny, nx), dtype=np.float32)
    dims = (np.int32(nz), np.int32(ny), np.int32(nx))
    _launch("rtt_vinterp", out.size,
            (cp.ascontiguousarray(mosaic), height_agl_m,
             cp.ascontiguousarray(terrain_m), cp.asarray(levels), out,
             np.int32(nlevels)) + dims)
    _launch("rtt_vinterp_edges", nz * (2 * (nx - 2) + 2 * (ny - 2) + 3),
            (out,) + dims)
    return out


def _smooth_weights(weight: float):
    """``smooth.f90:53-57`` in binary64, in the order the source states."""
    s = float(weight)
    return (0.25 * s * s, 0.5 * s * (1.0 - s), (1.0 - s) * (1.0 - s),
            (1.0 - s), 0.5 * s)


def smooth(field, passes: int = 1, weight: float = 0.5):
    """NOAA's ``SMOOTH`` applied ``passes`` times to every level of a
    float64 device volume ``(nz, ny, nx)``.  Returns a new array."""
    cp = _require_device()
    if (not isinstance(field, cp.ndarray) or field.dtype != np.float64
            or field.ndim != 3):
        raise RadarTtenError("smooth wants a float64 device (nz, ny, nx)")
    _require_extent((4,) + tuple(field.shape[1:]))
    current = cp.ascontiguousarray(field).copy()
    other = cp.empty_like(current)
    for _ in range(int(passes)):
        current, other = _smooth_pass(current, other, weight)
    return current


def _smooth_pass(src, dst, weight):
    nz, ny, nx = src.shape
    s1, s2, s3, s4, s5 = (np.float64(v) for v in _smooth_weights(weight))
    dims = (np.int32(nz), np.int32(ny), np.int32(nx))
    _launch("rtt_smooth_interior", src.size, (src, dst, s1, s2, s3) + dims)
    _launch("rtt_smooth_edges", nz * 4, (dst, s4, s5) + dims)
    return dst, src


def build_tendency(ref, theta, pressure_hpa, qv, height_agl_m,
                   config: RadarTtenConfig | None = None, *,
                   intermediates: dict | None = None):
    """One tendency slot from reflectivity and a background.

    All five inputs are float32 device volumes ``(nz, ny, nx)``:

    ``ref``
        reflectivity in NOAA's convention (module docstring); not modified.
    ``theta``
        potential temperature, K.
    ``pressure_hpa``
        pressure, hPa.
    ``qv``
        water vapour mixing ratio, kg/kg.
    ``height_agl_m``
        height above the ground, m.

    Returns ``(slot, receipt)``: the float32 device slot in NOAA's file
    convention and the build receipt (:func:`tendency_receipt`).  The call
    order is the driver's: boundary-layer top, cone fill, then
    ``radar_ref2tten``.  ``intermediates``, when given, receives the
    boundary-layer top and the cone-filled reflectivity (device arrays), for
    the oracle comparison.
    """
    cp = _require_device()
    cfg = RadarTtenConfig() if config is None else config
    ref = _volume(cp, "ref", ref)
    shape = ref.shape
    _require_extent(shape)
    theta = _volume(cp, "theta", theta, shape)
    pressure_hpa = _volume(cp, "pressure_hpa", pressure_hpa, shape)
    qv = _volume(cp, "qv", qv, shape)
    height_agl_m = _volume(cp, "height_agl_m", height_agl_m, shape)
    nz, ny, nx = (int(v) for v in shape)
    ncol = ny * nx
    dims = (np.int32(nz), np.int32(ny), np.int32(nx))

    pblh = pbl_height(qv, theta)
    refc = cone_fill(ref, height_agl_m, pblh)
    if intermediates is not None:
        intermediates["pblh"] = pblh
        intermediates["ref_cone"] = refc

    probable = cp.zeros((ny, nx), dtype=np.int8)
    if cfg.convection_only:
        _launch("rtt_probable_convection", ncol,
                (refc, pressure_hpa, theta, probable,
                 np.float64(cfg.convection_refl_threshold_dbz),
                 np.float64(cfg.warm_temperature_k),
                 np.float64(cfg.cold_echo_depth_hpa),
                 np.float64(_RD_OVER_CP)) + dims)

    extended = cp.zeros((ny, nx), dtype=np.int8)
    if cfg.pbl_extension:
        _launch("rtt_observed_depth", ncol,
                (refc, pressure_hpa, extended,
                 np.float64(cfg.pbl_extension_depth_hpa)) + dims)

    tten = cp.empty(shape, dtype=np.float64)
    work = cp.empty(shape, dtype=np.float64)
    _launch("rtt_tendency", tten.size,
            (refc, pressure_hpa, theta, pblh, extended, tten,
             np.float32(cfg.krad_bot), np.float64(cfg.echo_floor_dbz),
             np.float64(cfg.warm_temperature_k),
             np.float64(cfg.warm_min_dbz), np.float64(cfg.z_scale_dbz),
             np.float64(cfg.z_divisor), np.float64(cfg.z_factor),
             np.float64(1.0 / _CPOVR_P), np.float64(_LV_P + _LF0_P),
             np.float64(cfg.latent_heat_period_min), np.float64(_CPD_P),
             np.float64(cfg.tendency_cap), np.float64(_RD_OVER_CP)) + dims)
    for _ in range(int(cfg.smooth_passes_tendency)):
        tten, work = _smooth_pass(tten, work, cfg.smooth_weight)

    slot = cp.empty(shape, dtype=np.float32)
    _launch("rtt_store", slot.size, (tten, refc, slot, np.int64(slot.size)))

    for _ in range(int(cfg.smooth_passes_flag)):
        tten, work = _smooth_pass(tten, work, cfg.smooth_weight)
    _launch("rtt_column_flag", ncol,
            (tten, refc, pressure_hpa, pblh, extended, probable, slot,
             np.float32(cfg.krad_bot), np.float64(cfg.coverage_depth_hpa),
             np.float64(cfg.nearby_tendency),
             np.int32(1 if cfg.convection_only else 0)) + dims)
    del tten, work
    before_strict = None
    if cfg.strict_suppression:
        before_strict = suppression_receipt(slot, refc, cfg)
        _launch("rtt_strict_clear", slot.size - ncol,
                (slot, refc, np.float64(cfg.echo_floor_dbz),
                 np.int64(slot.size - ncol)))
    receipt = tendency_receipt(slot)
    if before_strict is not None:
        receipt["suppression_before_strict"] = before_strict
    receipt["probable_convection_columns"] = int(cp.count_nonzero(probable))
    receipt["boundary_layer_top_level"] = {
        "min": float(pblh.min()), "max": float(pblh.max())}
    receipt["cone_changed_points"] = int(cp.count_nonzero(refc != ref))
    receipt["pbl_extended_columns"] = int(cp.count_nonzero(extended))
    receipt["suppression"] = suppression_receipt(slot, refc, cfg)
    receipt["config"] = config_record(cfg)
    return slot, receipt


def config_record(config: RadarTtenConfig) -> dict:
    """Every constant a slot was built with, and whether it is NOAA's."""
    import dataclasses

    record = {f.name: getattr(config, f.name)
              for f in dataclasses.fields(config)}
    record["slot_minutes"] = list(record["slot_minutes"])
    record["noaa_product"] = config.is_noaa()
    return record


def suppression_receipt(slot, ref, config: RadarTtenConfig | None = None
                        ) -> dict:
    """The suppression rule, checked on a built slot (design E3).

    ``ref`` is the reflectivity the slot was built from AFTER the cone fill
    (the builder's ``intermediates["ref_cone"]``).  Levels below the top are
    classed by what the radar said there and what the slot then does:

    * observed no echo (``-100 < Z < echo_floor``): the rule is "no latent
      heating".  ``observed_clear_heated`` counts the points that still
      carry a non-zero tendency; NOAA's smoothing puts them there
      (:attr:`RadarTtenConfig.strict_suppression` removes them), and
      ``observed_clear_to_model`` counts points the column rules handed back
      to the microphysics;
    * echo: heated, held at zero (warm echo under ``warm_min_dbz``, below
      the heating's bottom, or a no-convection column) or handed to the
      model (stratiform columns);
    * ``holds``: no observed-clear point is heated or handed to the model.
    """
    cp = _require_device()
    cfg = RadarTtenConfig() if config is None else config
    below = slot[:-1]
    r = ref[:-1]
    covered = (below >= np.float32(-1.0)) & (below <= np.float32(1.0))
    nonzero = covered & (below != np.float32(0.0))
    clear = (r > np.float32(-100.0)) & (r < np.float32(cfg.echo_floor_dbz))
    echo = r >= np.float32(cfg.echo_floor_dbz)
    count = lambda m: int(cp.count_nonzero(m))          # noqa: E731
    clear_heated = clear & nonzero
    out = {
        "observed_clear_points": count(clear),
        "observed_clear_zero": count(clear & covered & (below == 0)),
        "observed_clear_heated": count(clear_heated),
        "observed_clear_to_model": count(clear & ~covered),
        "observed_clear_max_k_per_s": (
            float(cp.where(clear_heated, below, np.float32(0)).max())
            if count(clear_heated) else 0.0),
        "echo_points": count(echo),
        "echo_heated": count(echo & nonzero),
        "echo_zero": count(echo & covered & (below == 0)),
        "echo_to_model": count(echo & ~covered),
    }
    out["holds"] = (out["observed_clear_heated"] == 0
                    and out["observed_clear_to_model"] == 0)
    return out


def tendency_receipt(slot) -> dict:
    """What a slot holds, counted on the device.

    ``points_heated`` / ``points_zero_tendency`` / ``points_no_coverage``
    partition the levels below the top; ``max_tendency_k_per_s`` is over the
    covered ones; ``flag_columns`` counts the top level's three values.
    """
    cp = _require_device()
    below = slot[:-1]
    flag = slot[-1]
    covered = (below >= np.float32(-1.0)) & (below <= np.float32(1.0))
    covered_count = int(cp.count_nonzero(covered))
    zero = int(cp.count_nonzero(below == np.float32(0.0)))
    no_coverage = int(cp.count_nonzero(
        below == np.float32(NO_COVERAGE_TENDENCY)))
    largest = (float(cp.where(covered, below, np.float32(0.0)).max())
               if covered_count else 0.0)
    return {
        "shape": [int(v) for v in slot.shape],
        "points_below_top": int(below.size),
        "points_heated": covered_count - zero,
        "points_zero_tendency": zero,
        "points_no_coverage": no_coverage,
        "points_other": int(below.size) - covered_count - no_coverage,
        "max_tendency_k_per_s": largest,
        "flag_columns": {
            "no_information": int(cp.count_nonzero(flag == np.float32(-10.0))),
            "no_convection": int(cp.count_nonzero(flag == np.float32(0.0))),
            "convection_nearby": int(cp.count_nonzero(
                flag == np.float32(1.0))),
        },
    }


# --------------------------------------------------------------------------
# adapters: the observation document and the model state
# --------------------------------------------------------------------------

def reflectivity_from_document(document, *, z_source: str = "z_obs"):
    """Reflectivity in NOAA's convention from a ``gpuwm-obs.radar-grid``
    document: ``(ref, provenance)``, ``ref`` a float32 device volume.

    ``z_mask`` true is echo, at the file's dBZ.  A clear-air mask true is
    "the radar looked and found nothing", -99.  Neither is no coverage,
    -99999.  A file with no clear-air assessment yields echo and no coverage
    only, which heats observed storms and suppresses nothing; the provenance
    says so.  A file whose clear-air zeroes were established in a regime
    outside the allow-list is refused: a range-folded gate may be a storm,
    and reading it as clear would withhold heating from it.
    """
    cp = _require_device()
    variables = document["variables"]
    for name in (z_source, "z_mask"):
        if name not in variables:
            raise RadarTtenError(
                f"the observation document carries no {name!r}")
    z_obs = cp.asarray(np.ascontiguousarray(
        np.asarray(variables[z_source], dtype=np.float32)))
    z_mask = cp.asarray(np.ascontiguousarray(
        np.asarray(variables["z_mask"]) != 0).astype(np.int8))
    if z_obs.ndim != 3 or z_mask.shape != z_obs.shape:
        raise RadarTtenError(
            f"{z_source} {z_obs.shape} and z_mask {z_mask.shape} are not "
            "one (level, south_north, west_east) grid")
    unusable = int(cp.count_nonzero((z_mask != 0) & ~cp.isfinite(z_obs)))
    if unusable:
        raise RadarTtenError(
            f"{unusable} cell(s) are marked as reflectivity observations "
            "and hold no finite value; heating from them would be undefined")
    has_clear = "z0_mask" in variables
    source = document.get("clear_air_source") if has_clear else None
    if has_clear:
        if source not in _CLEAR_AIR_SOURCES:
            raise RadarTtenError(
                f"the file declares clear_air_source {source!r}; this "
                f"adapter reads {sorted(_CLEAR_AIR_SOURCES)}. How a zero "
                "was established decides whether it is an observation of "
                "nothing or a return the radar could not place, and a "
                "storm read as clear air loses its heating")
        z0_mask = cp.asarray(np.ascontiguousarray(
            np.asarray(variables["z0_mask"]) != 0).astype(np.int8))
        if z0_mask.shape != z_obs.shape:
            raise RadarTtenError(
                f"z0_mask {z0_mask.shape} is not on the grid of "
                f"{z_source} {z_obs.shape}")
        overlap = int(cp.count_nonzero((z0_mask != 0) & (z_mask != 0)))
        if overlap:
            raise RadarTtenError(
                f"{overlap} cell(s) are marked both as reflectivity "
                "observations and as clear air; the two halves of the file "
                "disagree about the same cell and no reading of it is safe")
    else:
        z0_mask = z_mask                      # never read: has_clear is 0
    ref = cp.empty(z_obs.shape, dtype=np.float32)
    _launch("rtt_adapt", ref.size,
            (z_obs, z_mask, z0_mask, ref, np.int32(1 if has_clear else 0),
             np.int64(ref.size)))
    provenance = {
        "z_source": z_source,
        "echo_points": int(cp.count_nonzero(z_mask)),
        "clear_air_points": (int(cp.count_nonzero(z0_mask))
                             if has_clear else 0),
        "clear_air": ("read" if has_clear else
                      "absent: the file has no clear-air assessment, so "
                      "this slot heats observed echo and suppresses nothing"),
        "clear_air_source": source,
    }
    return ref, provenance


def background_from_state(state) -> dict:
    """The builder's background from a live model state, on the device.

    ``theta = thb + thp``; pressure is the model's full pressure in hPa;
    vapour is ``qv``; height above ground is the half-level geopotential
    height minus the surface's, built as the microphysics prep builds it
    (``woof/core/microphysics.py``, the Kessler adapter).  NOAA's program
    instead rebuilds pressure and height hydrostatically from the surface
    pressure (``BackgroundCld.f90``); the model state is the authority here.
    """
    cp = _require_device()
    from woof.core import constants as c

    thp = state.thp
    if not isinstance(thp, cp.ndarray):
        raise RadarTtenError(
            "the model state is not on a CUDA device; the builder has no "
            "host path")
    if getattr(state, "qv", None) is None:
        raise RadarTtenError(
            "the model state carries no water vapour; the boundary-layer "
            "top needs virtual potential temperature")
    shape = thp.shape
    nz, ny, nx = (int(v) for v in shape)
    thb, phb = state.thb, state.phb
    theta = cp.empty(shape, dtype=np.float32)
    p_hpa = cp.empty(shape, dtype=np.float32)
    h_agl = cp.empty(shape, dtype=np.float32)
    _launch("rtt_background", theta.size,
            (cp.ascontiguousarray(thb), cp.ascontiguousarray(thp),
             cp.ascontiguousarray(state.p), cp.ascontiguousarray(phb),
             cp.ascontiguousarray(state.php), theta, p_hpa, h_agl,
             np.float32(c.G), np.int32(1 if thb.ndim == 3 else 0),
             np.int32(1 if phb.ndim == 3 else 0),
             np.int32(nz), np.int32(ny * nx)))
    return {"theta": theta, "pressure_hpa": p_hpa,
            "qv": cp.ascontiguousarray(state.qv), "height_agl_m": h_agl}


# --------------------------------------------------------------------------
# the model side
# --------------------------------------------------------------------------

def wrf_minutes(elapsed_seconds: float) -> np.float32:
    """Minutes since the start as WRF forms ``xtime``: float32, whole
    seconds over 60 plus the fraction over 60
    (``frame/module_domain.F:2316-2321``)."""
    whole = int(np.floor(elapsed_seconds))
    fraction = float(elapsed_seconds) - whole
    days, seconds = divmod(whole, 86400)
    minutes = np.float32(np.float32(days) * np.float32(24.0)
                         * np.float32(60.0)
                         + np.float32(seconds) / np.float32(60.0))
    if fraction != 0.0:
        minutes = np.float32(minutes + np.float32(fraction)
                             / np.float32(60.0))
    return minutes


def select_slot(minutes, slot_minutes) -> int:
    """The slot WRF reads at ``minutes``, as a ZERO-based index.

    ``module_big_step_utilities_em.F:5917-5921``: start at the first slot
    and move on while the clock is PAST the slot's time; the last slot
    persists.  ``minutes`` is the time at the beginning of the step: WRF
    advances ``xtime`` after the solver returns
    (``frame/module_integrate.F:366-369``, ``frame/module_domain.F:2550``).
    """
    times = tuple(np.float32(v) for v in slot_minutes)
    if not times:
        raise RadarTtenError("a forcing needs at least one slot")
    now = np.float32(minutes)
    index = 0
    while index < len(times) - 1 and now > times[index]:
        index += 1
    return index


def _init_activity(forcing, active_minutes) -> None:
    """``active_minutes``: the forcing applies while the clock (minutes
    from attachment, at the start of the step) is below it and is a
    pass-through after, so one integration can be forced for its first
    hour (a pre-forecast) and run free after it.  ``None``: always."""
    if active_minutes is not None:
        active_minutes = float(active_minutes)
        if not np.isfinite(active_minutes) or active_minutes <= 0.0:
            raise RadarTtenError(
                f"active_minutes {active_minutes}: a forcing that is never "
                "active would record a forced run that was not")
    forcing.active_minutes = active_minutes
    forcing.calls_after_active = 0
    forcing._inactive = False


def slab_extent(extent, shape):
    """``None`` (a whole domain) or ``(j0, i0, ny_domain, nx_domain)`` for a
    card's slab of shape ``(nz, ny, nx)``, checked to lie inside the domain.
    Host-only arithmetic, so the refusal is testable without a device."""
    if extent is None:
        return None
    try:
        j0, i0, gny, gnx = (int(v) for v in extent)
    except (TypeError, ValueError) as error:
        raise RadarTtenError(
            f"slab extent {extent!r} is not (j0, i0, ny, nx)") from error
    _, ny, nx = (int(v) for v in shape)
    if j0 < 0 or i0 < 0 or j0 + ny > gny or i0 + nx > gnx:
        raise RadarTtenError(
            f"a {ny} x {nx} slab at (j0={j0}, i0={i0}) does not lie inside "
            f"the {gny} x {gnx} domain: the tendency would be read at the "
            "wrong columns")
    return (j0, i0, gny, gnx)


def slab_refusal(forcing_extent, state_origin):
    """The reason a forcing with ``forcing_extent`` cannot be read on a
    state whose slab origin is ``state_origin`` (``None`` for a whole-domain
    state), or ``None`` when it can."""
    if forcing_extent is None:
        if state_origin is not None:
            return ("this state is a card's slab of a larger domain and the "
                    "forcing is a whole-domain field: the tendency would be "
                    "read at the slab's own (0, 0)")
        return None
    origin = tuple(forcing_extent[:2])
    if state_origin is None:
        return (f"the forcing is a slab at (j0, i0) = {origin} and this "
                "state is not a resident rank slab: a tile-streamed buffer "
                "serves a different tile every sweep")
    if tuple(state_origin) != origin:
        return (f"the forcing is a slab at (j0, i0) = {origin} and this "
                f"state is the slab at {tuple(state_origin)}: another "
                "rank's slab holds other columns")
    return None


def _ended(forcing) -> bool:
    return (forcing.active_minutes is not None
            and forcing.minutes() >= np.float32(forcing.active_minutes))


class RadarTtenForcing:
    """Tendency slots, their end times, and the clock that picks between
    them.  External data: it is attached to a state for one leg and is in
    no restart, device or state inventory.

    ``slots`` are float32 device volumes ``(nz, ny, nx)`` in NOAA's file
    convention; ``slot_minutes`` are their end times in minutes from the
    moment of attachment.  The elapsed-time counter starts at zero and is
    advanced by each microphysics call's ``dt``.

    ``mp_tend_lim`` is the microphysics heating clamp, K/s, the scheme runs
    under while the forcing is attached: HRRR's 0.07 by default (see
    :data:`HRRR_MP_TEND_LIM`), or ``None`` to keep the case's own.
    """

    def __init__(self, slots, slot_minutes, *, receipts=(), provenance=None,
                 mp_tend_lim: float | None = HRRR_MP_TEND_LIM,
                 active_minutes: float | None = None, extent=None):
        cp = _require_device()
        _init_activity(self, active_minutes)
        if mp_tend_lim is not None:
            mp_tend_lim = float(mp_tend_lim)
            if not np.isfinite(mp_tend_lim) or mp_tend_lim <= 0.0:
                raise RadarTtenError(
                    f"mp_tend_lim {mp_tend_lim} K/s: the clamp must be a "
                    "finite positive heating rate, or None to keep the "
                    "case's; zero would remove all microphysics heating "
                    "where the radar does not cover")
        self.mp_tend_lim = mp_tend_lim
        #: The case's clamp as the first microphysics call saw it.
        self.case_mp_tend_lim = None
        self._scheme_cfg = None
        slots = tuple(slots)
        times = tuple(float(v) for v in slot_minutes)
        if not slots or len(slots) != len(times):
            raise RadarTtenError(
                f"{len(slots)} slot(s) and {len(times)} slot time(s); a "
                "forcing needs one end time per slot and at least one slot")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise RadarTtenError(
                f"slot times {times} must increase: the slot rule walks "
                "them in order")
        shape = tuple(slots[0].shape)
        self.slots = tuple(_volume(cp, f"slot {index + 1}", slot, shape)
                           for index, slot in enumerate(slots))
        _require_extent(shape)
        self.shape = shape
        self.extent = slab_extent(extent, shape)
        self.slot_minutes = times
        self.receipts = tuple(dict(r) for r in receipts)
        self.provenance = dict(provenance or {})
        #: Seconds since attachment, at the beginning of the next step.
        self.elapsed_seconds = 0.0
        #: Microphysics calls each slot was applied on.
        self.calls_by_slot = [0] * len(self.slots)
        #: Calls on which ``no_mp_heating`` skipped the application.
        self.calls_skipped_no_mp_heating = 0
        self._theta_before = cp.empty(shape, dtype=np.float32)
        self._pending = None

    @classmethod
    def from_fields(cls, fields, slot_minutes, **kwargs):
        """A forcing from fields already in NOAA's file convention, for
        example ``RAD_TTEN_DFI_1..4`` and ``TTEN_TIMES`` of a start file.
        Host arrays are uploaded; nothing is converted."""
        cp = _require_device()
        slots = [cp.asarray(np.ascontiguousarray(
            np.asarray(field, dtype=np.float32)))
            if not isinstance(field, cp.ndarray) else field
            for field in fields]
        return cls(slots, slot_minutes, **kwargs)

    # -- the clock ---------------------------------------------------------
    def minutes(self) -> np.float32:
        return wrf_minutes(self.elapsed_seconds)

    def slot_index(self) -> int:
        return select_slot(self.minutes(), self.slot_minutes)

    # -- refusals ------------------------------------------------------------
    def check_state(self, state, cfg=None) -> None:
        """Refuse a state this forcing cannot be read on.

        The breakage each refusal prevents is the same one: a whole-domain
        tendency read at the wrong place, or not read at all, with no error.
        """
        cp = _require_device()
        if callable(getattr(type(state), "member_view", None)):
            raise RadarTtenError(
                "this is a batched-ensemble state: its members share one "
                "set of arrays in a member-major layout and the batched "
                "microphysics runs its own column adapter, so a "
                "whole-domain tendency would be read at the wrong offsets "
                "or not at all")
        if getattr(state, "_streamed_domain", None) is not None:
            raise RadarTtenError(
                "this state is integrated tile by tile (tile-streamed or "
                "multi-card): each tile or card slab runs microphysics on "
                "its own buffer, so a forcing attached to the whole-domain "
                "state would be skipped or read at the wrong offsets.  A "
                "multi-card run attaches one slab forcing per resident "
                "rank instead (woof.da.forecast_heating)")
        refusal = slab_refusal(self.extent,
                               getattr(state, SLAB_ATTRIBUTE, None))
        if refusal is not None:
            raise RadarTtenError(refusal)
        thp = getattr(state, "thp", None)
        if not isinstance(thp, cp.ndarray):
            raise RadarTtenError(
                "the model state is not on a CUDA device; the forcing has "
                "no host path")
        if tuple(thp.shape) != self.shape or thp.dtype != np.float32 \
                or not thp.flags.c_contiguous:
            raise RadarTtenError(
                f"the forcing is a whole-domain {self.shape} float32 field "
                f"and this state's theta is {thp.dtype} {tuple(thp.shape)}. "
                "A batched-ensemble column layout, a tile buffer or a card "
                "slab holds a different extent, and reading the tendency "
                "at its indices would heat the wrong columns")
        if cfg is not None and int(getattr(cfg, "mp_physics", 0)) == 0:
            raise RadarTtenError(
                "radar latent heating replaces the microphysics heating "
                "increment, and mp_physics = 0 runs no microphysics step: "
                "the tendency would never be applied and the slot clock "
                "would never advance")

    # -- the clamp the scheme runs under -------------------------------------
    def scheme_config(self, cfg):
        """``cfg`` with this forcing's ``mp_tend_lim``: what the scheme runs
        under while the forcing is attached.

        One replaced configuration per case configuration, kept for the
        leg, so a step costs an identity check.  ``None`` (keep the case's
        clamp) and a case already at the forcing's value return ``cfg``
        itself.
        """
        if self.case_mp_tend_lim is None:
            self.case_mp_tend_lim = float(cfg.mp_tend_lim)
        if self._inactive or self.mp_tend_lim is None \
                or float(cfg.mp_tend_lim) == self.mp_tend_lim:
            return cfg
        cached = self._scheme_cfg
        if cached is None or cached[0] is not cfg:
            import dataclasses

            cached = (cfg, dataclasses.replace(
                cfg, mp_tend_lim=self.mp_tend_lim))
            self._scheme_cfg = cached
        return cached[1]

    # -- the two halves of one microphysics call -----------------------------
    def before_microphysics(self, state, cfg, dt: float) -> None:
        """Keep theta as it is before the scheme, and pick this step's slot
        from the time at the beginning of the step."""
        cp = _require_device()
        self.check_state(state, cfg)
        self._inactive = _ended(self)
        if self._inactive:
            self._pending = -1
            return
        cp.copyto(self._theta_before, state.thp)
        self._pending = self.slot_index()

    def after_microphysics(self, state, cfg, dt: float, *,
                           ring_width: int = 0) -> None:
        """Replace the microphysics increment on covered points, then
        advance the clock.

        With ``no_mp_heating = 1`` WRF skips the whole block that holds the
        radar lines (``module_big_step_utilities_em.F:5949``), so nothing is
        applied; the call is counted so a receipt shows it.
        """
        if self._pending is None:
            raise RadarTtenError(
                "after_microphysics was called without before_microphysics")
        index, self._pending = self._pending, None
        if index == -1:
            self.calls_after_active += 1
            self.elapsed_seconds += float(dt)
            return
        if int(getattr(cfg, "no_mp_heating", 0)) == 0:
            nz, ny, nx = self.shape
            j0, i0, gny, gnx = (self.extent if self.extent is not None
                                else (0, 0, ny, nx))
            _launch("rtt_apply", self._theta_before.size,
                    (state.thp, self._theta_before, self.slots[index],
                     np.float32(dt), np.int32(max(int(ring_width), 0)),
                     np.int32(nz), np.int32(ny), np.int32(nx),
                     np.int32(j0), np.int32(i0), np.int32(gny),
                     np.int32(gnx)))
            self.calls_by_slot[index] += 1
        else:
            self.calls_skipped_no_mp_heating += 1
        self.elapsed_seconds += float(dt)

    # -- the record ----------------------------------------------------------
    def receipt(self) -> dict:
        return {
            "schema": "gpuwm-da.radar-tten-forcing.v1",
            "slot_minutes": list(self.slot_minutes),
            "slots": [dict(r) for r in self.receipts],
            "provenance": dict(self.provenance),
            "elapsed_seconds": float(self.elapsed_seconds),
            "calls_by_slot": list(self.calls_by_slot),
            "calls_skipped_no_mp_heating": int(
                self.calls_skipped_no_mp_heating),
            "active_minutes": self.active_minutes,
            "calls_after_active": int(self.calls_after_active),
            "mp_tend_lim_k_per_s": (
                self.mp_tend_lim if self.mp_tend_lim is not None
                else self.case_mp_tend_lim),
            "mp_tend_lim_source": (
                "the case's" if self.mp_tend_lim is None else
                "HRRR's, paired with mp_tend_radar = 1 "
                "(parm/conus/hrrr_wrfpre.nl:108-109)"
                if self.mp_tend_lim == HRRR_MP_TEND_LIM else
                "the forcing's, set by its caller"),
            "case_mp_tend_lim_k_per_s": self.case_mp_tend_lim,
            "extent": None if self.extent is None else list(self.extent),
        }


def attach(state, forcing: RadarTtenForcing, cfg=None) -> None:
    """Attach ``forcing`` to ``state`` for the coming leg.

    The forcing is external data, like boundary data: restart INFRA
    (``woof.io.restart.STATE_INFRA_ATTRS``), so a checkpoint written while
    it is attached carries none of it.  Its slot clock starts at zero here,
    so a caller that resumes inside a forced period must refuse
    (``woof.da.forecast_heating.resume_plan``) rather than re-attach.
    """
    if not isinstance(forcing, (RadarTtenForcing, LatentHeatNudging)):
        raise RadarTtenError(
            "expected a RadarTtenForcing or LatentHeatNudging, got "
            f"{type(forcing).__name__}")
    if getattr(state, STATE_ATTRIBUTE, None) is not None:
        raise RadarTtenError(
            "this state already carries a radar forcing; detach it first, "
            "or two legs' tendencies would share one clock")
    forcing.check_state(state, cfg)
    setattr(state, STATE_ATTRIBUTE, forcing)


def detach(state):
    """Remove and return the state's forcing (``None`` if it had none)."""
    forcing = getattr(state, STATE_ATTRIBUTE, None)
    if STATE_ATTRIBUTE in vars(state):
        delattr(state, STATE_ATTRIBUTE)
    return forcing


def build_forcing_from_documents(state, documents, slot_minutes,
                                 config: RadarTtenConfig | None = None, *,
                                 mp_tend_lim: float | None = HRRR_MP_TEND_LIM):
    """One slot per observation document, each built against ``state``.

    This is the pre-forecast shape: NOAA builds all four slots against the
    one background the hour starts from, each from the observations valid at
    the END of its window.  ``slot_minutes`` are the windows' end times in
    minutes from now.
    """
    refs, sources = [], []
    for document in documents:
        ref, source = reflectivity_from_document(document)
        refs.append(ref)
        sources.append(source)
    return build_forcing(
        state, refs, slot_minutes, config, mp_tend_lim=mp_tend_lim,
        sources=sources,
        provenance={"built_from": "gpuwm-obs.radar-grid documents",
                    "background": "the model state at attachment"})


def build_forcing(state, refs, slot_minutes,
                  config: RadarTtenConfig | None = None, *,
                  mp_tend_lim: float | None = HRRR_MP_TEND_LIM,
                  sources=None, provenance=None,
                  active_minutes: float | None = None):
    """One slot per reflectivity volume (NOAA's convention, device float32
    ``(nz, ny, nx)``), each built against ``state`` as it stands now.

    The pre-forecast and leg shape: NOAA builds all four slots against the
    one background the hour starts from (``gsdcloudanalysis_ref2tten.f90``),
    each from the volume valid at the END of its window, and the slot rule
    (:func:`select_slot`) reads window ``n`` while the clock is inside it.
    """
    refs = list(refs)
    if len(refs) != len(tuple(slot_minutes)):
        raise RadarTtenError(
            f"{len(refs)} reflectivity volume(s) for "
            f"{len(tuple(slot_minutes))} slot time(s): each window needs "
            "its own volume, or a window would be heated from another "
            "window's storms")
    background = background_from_state(state)
    shape = tuple(getattr(background["theta"], "shape", ()))
    slots, receipts = [], []
    for index, ref in enumerate(refs):
        if shape and tuple(getattr(ref, "shape", shape)) != shape:
            raise RadarTtenError(
                f"the observation grid {tuple(ref.shape)} is not the model "
                f"grid {shape}")
        slot, receipt = build_tendency(ref, config=config, **background)
        if sources is not None:
            receipt["observations"] = sources[index]
        slots.append(slot)
        receipts.append(receipt)
    del background
    return RadarTtenForcing(
        slots, slot_minutes, receipts=receipts,
        provenance=dict(provenance or {
            "built_from": "reflectivity volumes",
            "background": "the model state at attachment"}),
        mp_tend_lim=mp_tend_lim,
        **({} if active_minutes is None else
           {"active_minutes": active_minutes}))


# --------------------------------------------------------------------------
# windows and members (design E3: four 15-minute windows, per-member
# parameters)
# --------------------------------------------------------------------------

#: Windows per leg hour in HRRR's pre-forecast
#: (``gsdcloudanalysis_ref2tten.f90:438-441``).
WINDOWS_PER_HOUR = 4


def window_end_minutes(leg_minutes: float, windows: int) -> tuple:
    """End times, minutes from the leg start, of ``windows`` equal windows.

    A 60-minute leg in four windows is NOAA's ``(15, 30, 45, 60)``.  The
    volume for window ``n`` is the one valid at its end."""
    leg_minutes = float(leg_minutes)
    windows = int(windows)
    if windows < 1 or not np.isfinite(leg_minutes) or leg_minutes <= 0.0:
        raise RadarTtenError(
            f"{windows} window(s) over {leg_minutes} min: a forced leg needs "
            "at least one window of positive length")
    return tuple(leg_minutes * (n + 1) / windows for n in range(windows))


#: Per-member parameter ranges (design E3, risk 2: "heating too strong
#: reproduces over-intensity"; NOAA moved dt_cond 5 -> 10 -> 20 min).
#: Each member draws uniformly inside these, around the base config.
MEMBER_PERTURBATION = {
    "latent_heat_period_min": (15.0, 30.0),
    "threshold_offset_dbz": (-3.0, 3.0),
}


def member_config(base: RadarTtenConfig | None, member: int, seed: int,
                  ranges: dict | None = None):
    """``(config, record)`` for one ensemble member: dt_cond and the 28 dBZ
    thresholds drawn per member from :data:`MEMBER_PERTURBATION`.

    Deterministic in ``(seed, member)`` alone, so a rerun, a resumed leg and
    a different card count draw the same values.  The control (member
    ``None``) keeps ``base`` unchanged.
    """
    import dataclasses

    base = RadarTtenConfig() if base is None else base
    if member is None:
        return base, {"member": None, "perturbed": False}
    ranges = dict(MEMBER_PERTURBATION if ranges is None else ranges)
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, int(member), 0x6C366])) # "l6"
    lo, hi = ranges["latent_heat_period_min"]
    period = float(rng.uniform(lo, hi))
    lo, hi = ranges["threshold_offset_dbz"]
    offset = float(rng.uniform(lo, hi))
    config = dataclasses.replace(
        base, latent_heat_period_min=period,
        convection_refl_threshold_dbz=base.convection_refl_threshold_dbz
        + offset,
        warm_min_dbz=base.warm_min_dbz + offset)
    return config, {"member": int(member), "perturbed": True,
                    "seed": int(seed),
                    "latent_heat_period_min": period,
                    "threshold_offset_dbz": offset,
                    "ranges": {k: list(v) for k, v in ranges.items()}}


def window_paths(root, leg_start_valid, leg_minutes: float,
                 windows: int) -> list:
    """The window products a forced leg reads, in slot order.

    ``root`` holds one directory per window end time,
    ``<root>/<YYYYmmddTHHMMZ>/ref.f32`` (``tools/radar_tten_windows.py``).
    A missing window is a refusal: heating three windows of four would
    leave a quarter of the leg on the microphysics while the record says
    forced."""
    from datetime import timedelta

    root = Path(root)
    paths, missing = [], []
    for end in window_end_minutes(leg_minutes, windows):
        valid = leg_start_valid + timedelta(minutes=float(end))
        path = root / f"{valid:%Y%m%dT%H%MZ}" / "ref.f32"
        (paths if path.is_file() else missing).append(path)
    if missing:
        raise RadarTtenError(
            f"{len(missing)} of {windows} window product(s) are missing "
            f"under {root}: {[str(p) for p in missing]}")
    return paths


def read_window_host(path, *, identity_sha256: str):
    """A window product held to its receipt and to the run's grid, on the
    host: ``(ref, receipt)``, ``ref`` float32 ``(nz, ny, nx)``.

    The strict reader (:func:`woof.obs.radar_tten_grid.read_reflectivity`)
    on every path: schema and READY status, a shape that accounts for every
    byte, the data digest, and ``grid.identity_sha256`` equal to the run's.
    ``identity_sha256`` is required.  The breakage it prevents: a window
    gridded onto another grid of the same shape was read and heated the
    wrong columns with no error (lane 6 code map, section 1).
    """
    from woof.obs.radar_tten_grid import RadarTtenGridError, read_reflectivity

    if not isinstance(identity_sha256, str) or not identity_sha256:
        raise RadarTtenError(
            "a heating window is read only against the run's grid identity; "
            "none was given, and an unchecked window may hold other columns")
    path = Path(path)
    if path.suffix != ".f32":
        raise RadarTtenError(f"{path} is not a window's ref.f32")
    try:
        host, receipt = read_reflectivity(path.with_suffix(""),
                                          identity_sha256=identity_sha256)
    except RadarTtenGridError as error:
        raise RadarTtenError(f"window {path}: {error}") from error
    named = Path(str((receipt.get("data") or {}).get("file")))
    named = named if named.is_absolute() else path.parent / named
    if named.resolve() != path.resolve():
        raise RadarTtenError(
            f"{path.with_suffix('.json')} names data {named}, not {path}")
    return host, receipt


def read_window_reflectivity(path, *, identity_sha256: str):
    """A ``gpuwm-obs.radar-tten-ref.v1`` window as ``(ref, receipt)``, ``ref``
    a float32 device volume, after :func:`read_window_host` has held it to
    its receipt and to ``identity_sha256``, the run's grid."""
    host, receipt = read_window_host(path, identity_sha256=identity_sha256)
    cp = _require_device()
    return cp.asarray(np.ascontiguousarray(host)), receipt


# --------------------------------------------------------------------------
# latent heat nudging: the comparison arm
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LatentHeatNudgingConfig:
    """ICON/COSMO-style latent heat nudging (design E3, comparison arm).

    The model's own microphysics heating at each step is scaled by the
    ratio of the radar's surface rain rate to the model's, inside
    ``[alpha_min, alpha_max]``, at heating points only, capped, and with
    relative humidity kept.  Declared simplifications against ICON: the
    model rate is the step's own grid-scale precipitation (no time filter),
    and a column where the radar sees rain and the model none is left alone
    (no upstream profile search)."""

    alpha_min: float = 0.5
    alpha_max: float = 2.0
    #: mm/h below which a rate counts as no rain.
    threshold_mm_h: float = 0.1
    #: K/h cap on the nudging increment.
    cap_k_per_h: float = 50.0
    #: Observed rate from the lowest observed level at or below this height.
    max_obs_height_m: float = 2500.0
    #: Marshall-Palmer Z = a R^b.
    zr_a: float = 200.0
    zr_b: float = 1.6


class LatentHeatNudging:
    """Observed rain-rate slots and the clock, read by
    ``woof.core.microphysics.apply`` exactly like :class:`RadarTtenForcing`
    (the same three calls), so either attaches as the state's forcing."""

    def __init__(self, rates, slot_minutes, *, config=None, receipts=(),
                 provenance=None, shape=None,
                 active_minutes: float | None = None):
        cp = _require_device()
        _init_activity(self, active_minutes)
        self.config = LatentHeatNudgingConfig() if config is None else config
        rates = tuple(rates)
        times = tuple(float(v) for v in slot_minutes)
        if not rates or len(rates) != len(times):
            raise RadarTtenError(
                f"{len(rates)} rate slot(s) and {len(times)} time(s)")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise RadarTtenError(f"slot times {times} must increase")
        self.rates = tuple(cp.ascontiguousarray(r.astype(np.float32))
                           for r in rates)
        self.shape = tuple(shape)
        #: Whole-domain only: the nudging arm has no slab form (its kernel
        #: tests the ring in the array's own indices), so a multi-card
        #: attach is refused by check_state's slab rule.
        self.extent = None
        self.slot_minutes = times
        self.receipts = tuple(dict(r) for r in receipts)
        self.provenance = dict(provenance or {})
        self.elapsed_seconds = 0.0
        self.calls_by_slot = [0] * len(self.rates)
        self.calls_skipped_no_mp_heating = 0
        self.points_up = 0
        self.points_down = 0
        self.mp_tend_lim = None
        self.case_mp_tend_lim = None
        self._theta_before = cp.empty(self.shape, dtype=np.float32)
        self._counts = cp.zeros(2, dtype=np.uint64)
        self._pending = None

    minutes = RadarTtenForcing.minutes

    slot_index = RadarTtenForcing.slot_index

    def check_state(self, state, cfg=None) -> None:
        RadarTtenForcing.check_state(self, state, cfg)

    def scheme_config(self, cfg):
        if self.case_mp_tend_lim is None:
            self.case_mp_tend_lim = float(cfg.mp_tend_lim)
        return cfg

    before_microphysics = RadarTtenForcing.before_microphysics

    def after_microphysics(self, state, cfg, dt: float, *,
                           ring_width: int = 0) -> None:
        if self._pending is None:
            raise RadarTtenError(
                "after_microphysics was called without before_microphysics")
        index, self._pending = self._pending, None
        if index == -1:
            self.calls_after_active += 1
            self.elapsed_seconds += float(dt)
            return
        if int(getattr(cfg, "no_mp_heating", 0)) == 0:
            nz, ny, nx = self.shape
            c = self.config
            thb = state.thb
            _launch("lhn_apply", self._theta_before.size,
                    (state.thp, self._theta_before, state.qv,
                     _require_device().ascontiguousarray(thb), state.p,
                     self.rates[index], state.scratch((ny, nx), "mp_rainncv"),
                     np.float32(dt), np.float32(c.alpha_min),
                     np.float32(c.alpha_max), np.float32(c.threshold_mm_h),
                     np.float32(c.cap_k_per_h / 3600.0),
                     np.int32(1 if thb.ndim == 3 else 0),
                     np.int32(max(int(ring_width), 0)), self._counts,
                     np.int32(nz), np.int32(ny), np.int32(nx)))
            self.calls_by_slot[index] += 1
        else:
            self.calls_skipped_no_mp_heating += 1
        self.elapsed_seconds += float(dt)

    def receipt(self) -> dict:
        import dataclasses

        up, down = (int(v) for v in self._counts.get())
        return {
            "schema": "gpuwm-da.latent-heat-nudging.v1",
            "config": dataclasses.asdict(self.config),
            "slot_minutes": list(self.slot_minutes),
            "slots": [dict(r) for r in self.receipts],
            "provenance": dict(self.provenance),
            "elapsed_seconds": float(self.elapsed_seconds),
            "calls_by_slot": list(self.calls_by_slot),
            "calls_skipped_no_mp_heating": int(
                self.calls_skipped_no_mp_heating),
            "active_minutes": self.active_minutes,
            "calls_after_active": int(self.calls_after_active),
            "point_steps_nudged_up": up,
            "point_steps_nudged_down": down,
            "mp_tend_lim_k_per_s": self.case_mp_tend_lim,
            "mp_tend_lim_source": "the case's (nudging scales the scheme's "
                                  "own heating; no companion clamp)",
        }


def build_nudging(state, refs, slot_minutes,
                  config: LatentHeatNudgingConfig | None = None, *,
                  sources=None, provenance=None,
                  active_minutes: float | None = None
                  ) -> LatentHeatNudging:
    """One observed rain-rate slot per reflectivity volume (NOAA's
    convention), from its lowest observed level at or below
    ``config.max_obs_height_m`` above the model's ground."""
    cp = _require_device()
    cfg = LatentHeatNudgingConfig() if config is None else config
    refs = list(refs)
    background = background_from_state(state)
    h_agl = background["height_agl_m"]
    shape = tuple(h_agl.shape)
    nz, ny, nx = shape
    del background
    rates, receipts = [], []
    for index, ref in enumerate(refs):
        ref = _volume(cp, f"reflectivity {index + 1}", ref, shape)
        rate = cp.empty((ny, nx), dtype=np.float32)
        _launch("lhn_observed_rate", ny * nx,
                (ref, h_agl, rate, np.float32(cfg.max_obs_height_m),
                 np.float32(cfg.zr_a), np.float32(cfg.zr_b),
                 np.int32(nz), np.int32(ny * nx)))
        receipt = {
            "columns_no_information": int(cp.count_nonzero(rate < 0)),
            "columns_observed_dry": int(cp.count_nonzero(
                (rate >= 0) & (rate < cfg.threshold_mm_h))),
            "columns_observed_rain": int(cp.count_nonzero(
                rate >= cfg.threshold_mm_h)),
            "max_rate_mm_h": float(rate.max()),
        }
        if sources is not None:
            receipt["observations"] = sources[index]
        rates.append(rate)
        receipts.append(receipt)
    return LatentHeatNudging(
        rates, slot_minutes, config=cfg, receipts=receipts, shape=shape,
        provenance=dict(provenance or {"built_from": "reflectivity volumes"}),
        active_minutes=active_minutes)
