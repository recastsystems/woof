"""WRF ``swint_opt = 1``: the surface shortwave between radiation calls.

Authority: WRF v4.6.1 ``phys/module_radiation_driver.F`` (commit d66e442,
sha256 77987565...), graded word for word by tools/swint_wrf461_oracle.
With ``swint_opt = 1`` the driver does three things:

* on EVERY step, before the radiation branch, ``radconst`` and
  ``calc_coszen`` at the CURRENT ``xtime`` fill ``coszen_loc``
  (:1158-1170; the radiation step keeps its own ``xtime + radt*0.5`` call
  at :1204-1206);
* on a radiation step, after the shortwave scheme wrote SWDOWN and SWDDIR,
  ``update_swinterp_parameters`` (:3543-3624) fits each column's direct
  and global surface flux as a power of the solar zenith cosine,
  ``SWDDIR = Bx * coszen**bb`` and ``SWDOWN = Gx * coszen**gg``, the
  exponents from the log ratio between this call and the stored reference
  (a linear first guess on a column whose ``Bx``/``Gx`` is not yet
  positive), clamped to [-0.5, 2.5], then stores this call as the
  reference (``coszen_ref``, ``swdown_ref``, ``swddir_ref``);
* on EVERY step, radiation steps included (the ``if (swint_opt .eq. 1)``
  block at :2964-2980 sits after ``ENDIF Radiation_step``),
  ``interp_sw_radiation`` (:3626-3670) rewrites SWDDIR, SWDOWN, SWDDIF,
  SWDDNI and GSW at ``coszen_loc``, a clamped exponent or an exponent of
  exactly zero falling back to the ratio ``coszen_loc/coszen_ref``, night
  columns (either cosine at or below 1e-4) zeroed.  GSW is
  ``SWDOWN * (1 - ALBEDO)`` with the LIVE albedo of the step: v4.6.1
  passes ``grid%albedo`` (module_first_rk_step_part1.F:265), which the
  land surface updates between radiation calls.  Under the fork-only
  sun-angle albedo (``alb_sol = 1``) the live field is ALBSOL, as the
  operational HRRR fork passes it.

``Bx``, ``bb``, ``Gx`` and ``gg`` are zeroed at ``itimestep == 1``
(:1708-1717); the three references are Registry ``misc`` state and start
at zero (Registry.EM_COMMON:1729-1731, all restart-carried).  None of the
fit touches the heating rates: RTHRATENSW stays the radiation call's.

The operational HRRR fork (NOAA-EMC/HRRR v4.1.21, WRFV3.9 base, commit
40ee6058c) carries these routines from before two upstream WRF bug fixes,
and the engine follows v4.6.1, which has both:

* WRF cc02f989f (v3.9.1, "fix for sunrise/sunset swint_opt calcs"): an
  exponent of exactly zero takes the ratio rule.  The fork evaluates
  ``Bx * coszen**0 = Bx`` there, so a column whose fit carried no sun
  angle information (a ratio of one, or both fluxes at or under 1 W m-2,
  where ``max(1, flux)`` makes both logs zero) held its flux flat until
  the next radiation call.
* WRF 650be0c7d (v4.3): ``calc_coszen`` bounds the cosine to [-1, 1]; the
  float32 sum can round one spacing past one at the subsolar point.

Everything the land surface reads between radiation calls therefore
follows the sun on every step, which is why HRRR runs it with a 15-minute
``radt``.  The engine held the radiation call's fluxes for the whole
interval before this module.

Numerics: the kernels in ``woof/core/kernels/swint.cu`` use WOOF's own
float32 ``logf``/``powf``/``sinf``/``cosf``/``asinf`` routines and
explicit single-rounded arithmetic, so they are bitwise WRF's Fortran
compiled with gfortran -O0 wherever the C library's float32 routines
return the correctly rounded word (tools/swint_wrf461_oracle records the
one exception it found: an older C library's ``asinf``).  The NumPy twins
below do the fit and the evaluation with the same words
(woof.core.noahmp_libm) and are bitwise the same; the twin of the
per-step zenith cosine (:func:`coszen_loc_host`) uses NumPy's
transcendentals, so it is the shape check, and the tests hold the
KERNEL's cosine to WRF's directly.

"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

F = np.float32

#: coszen_min in both Fortran routines.
COSZEN_MIN = F(1.0e-4)

#: The 2-D float32 fields swint_opt = 1 adds to the surface inventory, in
#: the driver's ``fields`` dict so the restart stream carries them in
#: place (woof/io/restart.py serializes every ``fields`` entry).  The
#: WRF names where WRF has one (swddir/swddni/swddif, the three *_ref),
#: a ``swint_`` prefix on the four fit coefficients WRF spells Bx, bb, Gx,
#: gg, and ``swint_albedo`` for the albedo the last radiation call used.
STATE_FIELDS = (
    "swddir", "swddif", "swddni",
    "swdown_ref", "swddir_ref", "coszen_ref",
    "swint_bx", "swint_bb", "swint_gx", "swint_gg",
    "swint_albedo",
)


# ---------------------------------------------------------------------------
# Calendar scalars at the CURRENT xtime.
# ---------------------------------------------------------------------------

def calendar_scalars(start_time: datetime, elapsed_seconds: float, *,
                     generation: str = "wrf_480"):
    """``(julian, xtime, gmt)`` float32 at ``elapsed_seconds``.

    ``wrf_480`` (the default) returns WRF's frame words
    (woof.core.solar.radiation_calendar); ``wrf_461`` the statements below.

    The same statements the legacy RRTMG adapter forms for its radiation
    call (woof/core/rrtmg_legacy.py ``RRTMGLegacyRadiation.__call__``,
    "calendar / solar"), so the per-step cosine and the radiation call's
    run on one calendar; the step's own ``xtime`` without the
    ``radt * 0.5`` the driver adds only for the radiation call.
    """
    from woof.core.solar import radiation_calendar
    if generation == "wrf_480":
        return radiation_calendar(start_time, elapsed_seconds, generation=generation)
    from woof.core.solar import solar_generation
    solar_generation(generation)
    valid_time = start_time + timedelta(seconds=float(elapsed_seconds))
    julday = int(valid_time.timetuple().tm_yday)
    hour = (valid_time.hour + valid_time.minute / 60.0
            + valid_time.second / 3600.0 + valid_time.microsecond / 3.6e9)
    julian = F((julday - 1) + hour / 24.0)
    gmt = F(start_time.hour + start_time.minute / 60.0
            + start_time.second / 3600.0 + start_time.microsecond / 3.6e9)
    xtime = F(float(elapsed_seconds) / 60.0)
    return julian, xtime, gmt


def coszen_loc_host(xlat, xlon, julian, xtime, gmt, *, generation="wrf_480"):
    """``radconst`` + ``calc_coszen`` per point (float32, NumPy libm).

    The statement order of ``swint_coszen_loc``; its transcendentals are
    NumPy's, so this twin is the shape check, and the kernel itself is
    held to the fork's words.
    """
    from woof.core.solar import solar_generation, solar_geometry_device
    solar_generation(generation)
    if generation == "wrf_480":
        import cupy as cp
        return cp.asnumpy(solar_geometry_device(julian, xtime, gmt, xlat, xlon, generation=generation)[1])
    from woof.core.rrtmg_legacy import _DEGRAD, radconst
    declin, _solcon = radconst(F(julian))
    xlat = np.asarray(xlat, np.float32)
    xlon = np.asarray(xlon, np.float32)
    j = F(julian)
    da = (F(6.2831853071795862) * (j - F(1.0))) / F(365.0)
    two = F(2.0) * da
    eot = F(0.000075) + F(0.001868) * np.cos(da)
    eot = eot - F(0.032077) * np.sin(da)
    eot = eot - F(0.014615) * np.cos(two)
    eot = eot - F(0.04089) * np.sin(two)
    eot = eot * F(229.18)
    xt24 = np.fmod(F(xtime), F(1440.0)) + eot
    tloctm = (F(gmt) + xt24 / F(60.0)) + xlon / F(15.0)
    hrang = (F(15.0) * (tloctm - F(12.0))) * _DEGRAD
    xxlat = xlat * _DEGRAD
    d = F(declin)
    coszen = (np.sin(xxlat) * np.sin(d)
              + (np.cos(xxlat) * np.cos(d)) * np.cos(hrang)).astype(np.float32)
    # v4.6.1 calc_coszen: coszen = min(max(coszen, -1.0), 1.0)
    return np.minimum(np.maximum(coszen, F(-1.0)), F(1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# NumPy twins of update_swinterp_parameters / interp_sw_radiation, with
# WOOF's own float32 logf and powf (woof.core.noahmp_libm), element by
# element.
# ---------------------------------------------------------------------------

def _logf(x):
    from woof.core.noahmp_libm import logf
    return logf(F(x))


def _powf(x, y):
    from woof.core.noahmp_libm import powf
    return powf(F(x), F(y))


def _exponent(flux, flux_0, coszen, coszen_0):
    """One fit exponent (the DIR and GHI halves are the same statements)."""
    flux, flux_0, coszen, coszen_0 = F(flux), F(flux_0), F(coszen), F(coszen_0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = F(coszen / coszen_0)
        if ratio < F(1.0):
            b = F(_logf(F(max(F(1.0), flux) / max(F(1.0), flux_0)))
                  / _logf(min(F(1.0) - F(1.0e-4), ratio)))
        elif ratio > F(1.0):
            b = F(_logf(F(max(F(1.0), flux) / max(F(1.0), flux_0)))
                  / _logf(max(F(1.0) + F(1.0e-4), ratio)))
        else:
            b = F(0.0)
    return F(max(F(-0.5), min(F(2.5), b)))


def update_swinterp_parameters(coszen, coszen_loc, swddir, swdown,
                               swddir_ref, bb, bx, swdown_ref, gg, gx,
                               coszen_ref):
    """``update_swinterp_parameters`` (:3543-3624) on float32 arrays.

    ``swddir_ref, bb, bx, swdown_ref, gg, gx, coszen_ref`` are updated IN
    PLACE, as the Fortran intent(inout) arguments are.
    """
    coszen = np.asarray(coszen, np.float32).reshape(-1)
    coszen_loc = np.asarray(coszen_loc, np.float32).reshape(-1)
    swddir = np.asarray(swddir, np.float32).reshape(-1)
    swdown = np.asarray(swdown, np.float32).reshape(-1)
    flat = [a.reshape(-1) for a in (swddir_ref, bb, bx, swdown_ref, gg, gx,
                                    coszen_ref)]
    swddir_ref_f, bb_f, bx_f, swdown_ref_f, gg_f, gx_f, coszen_ref_f = flat
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(coszen.size):
            cz, czl = coszen[i], coszen_loc[i]
            if cz > COSZEN_MIN and czl > COSZEN_MIN:
                if bx_f[i] <= F(0.0):
                    swddir_0 = F(F(czl / cz) * swddir[i])
                    coszen_0 = czl
                else:
                    swddir_0 = swddir_ref_f[i]
                    coszen_0 = coszen_ref_f[i]
                b = _exponent(swddir[i], swddir_0, cz, coszen_0)
                bb_f[i] = b
                bx_f[i] = F(swddir[i] / _powf(cz, b))
                if gx_f[i] <= F(0.0):
                    swdown_0 = F(F(czl / cz) * swdown[i])
                    coszen_0 = czl
                else:
                    swdown_0 = swdown_ref_f[i]
                    coszen_0 = coszen_ref_f[i]
                g = _exponent(swdown[i], swdown_0, cz, coszen_0)
                gg_f[i] = g
                gx_f[i] = F(swdown[i] / _powf(cz, g))
            else:
                bx_f[i] = bb_f[i] = gx_f[i] = gg_f[i] = F(0.0)
            coszen_ref_f[i] = cz
            swdown_ref_f[i] = swdown[i]
            swddir_ref_f[i] = swddir[i]


def _ratio_rule(exponent):
    """v4.6.1's test for the ratio fallback: a clamped exponent, or an
    exponent of exactly zero (WRF cc02f989f)."""
    return exponent == F(-0.5) or exponent == F(2.5) or exponent == F(0.0)


def interp_sw_radiation(coszen_ref, coszen_loc, swddir_ref, bb, bx,
                        swdown_ref, gg, gx, albedo):
    """``interp_sw_radiation`` (:3626-3670); returns
    ``(swdown, swddir, swddni, swddif, gsw)`` float32."""
    args = [np.asarray(a, np.float32).reshape(-1) for a in
            (coszen_ref, coszen_loc, swddir_ref, bb, bx, swdown_ref, gg, gx,
             albedo)]
    czr, czl, dref, bb, bx, sref, gg, gx, alb = args
    n = czr.size
    out = [np.zeros(n, np.float32) for _ in range(5)]
    swdown, swddir, swddni, swddif, gsw = out
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n):
            if czr[i] > COSZEN_MIN and czl[i] > COSZEN_MIN:
                if _ratio_rule(bb[i]):
                    d = F(F(czl[i] / czr[i]) * dref[i])
                else:
                    d = F(bx[i] * _powf(czl[i], bb[i]))
                if _ratio_rule(gg[i]):
                    s = F(F(czl[i] / czr[i]) * sref[i])
                else:
                    s = F(gx[i] * _powf(czl[i], gg[i]))
                swddir[i] = d
                swdown[i] = s
                swddif[i] = F(s - d)
                swddni[i] = F(d / czl[i])
                gsw[i] = F(s * F(F(1.0) - alb[i]))
    return swdown, swddir, swddni, swddif, gsw


# ---------------------------------------------------------------------------
# The driver-side carrier: device state and the two step halves.
# ---------------------------------------------------------------------------

_BLOCK = 256


def _host(value):
    """A host float32 copy of a scalar, NumPy or CuPy value."""
    if hasattr(value, "get") and hasattr(value, "device"):
        return np.asarray(value.get(), np.float32)
    return np.asarray(value, np.float32)


class ShortwaveInterpolation:
    """The ``swint_opt = 1`` half of WRF's radiation driver on one domain.

    Holds this domain's latitude and longitude on the device and the
    per-step ``coszen_loc`` scratch (a local in the Fortran, so never
    serialized: woof/io/restart.py classifies the object REBUILT and the
    eleven state fields ride the ``fields`` inventory).  Two entry points
    mirror the driver: :meth:`radiation_step` right after a radiation
    call captured its result, :meth:`between_calls` on every other step.

    The latitude and longitude are ``latitude_deg`` and ``longitude_deg``,
    (ny, nx) arrays under the names every scheme that owns its geography
    uses, and they are read at each call.  A streamed tile or a slab of a
    multi-card run is a buffer built on neutral geography whose scheme
    grids are gathered from the domain (tilestream.driver
    ``_SCHEME_GEOGRAPHY``, which lists this carrier): a private copy made
    at construction would keep the neutral constant and light every
    column at one sun.
    """

    def __init__(self, *, start_time: datetime, latitude_deg, longitude_deg,
                 shape, generation="wrf_480"):
        import cupy as cp
        from woof.core.kernels import load_module

        if not isinstance(start_time, datetime):
            raise TypeError("swint_opt = 1 needs the run's UTC start time "
                            "as a datetime")
        from woof.core.solar import solar_generation
        solar_generation(generation)
        self.solar_generation = generation
        self.start_time = start_time
        ny, nx = (int(shape[0]), int(shape[1]))
        self.shape = (ny, nx)
        self.ncol = ny * nx
        lat = np.ascontiguousarray(np.broadcast_to(
            _host(latitude_deg).astype(np.float32), self.shape))
        lon = np.ascontiguousarray(np.broadcast_to(
            _host(longitude_deg).astype(np.float32), self.shape))
        self.latitude_deg = cp.asarray(lat)
        self.longitude_deg = cp.asarray(lon)
        # (ny, nx), not flat: a leading ny*nx axis is the layout the tile
        # transport cannot window and refuses on a scheme object
        self._coszen_loc = cp.zeros(self.shape, dtype=cp.float32)
        module = load_module("swint")
        self._k_coszen = module.get_function("swint_coszen_loc")
        self._k_update = module.get_function("swint_update")
        self._k_interp = module.get_function("swint_interp")
        self._grid = ((self.ncol + _BLOCK - 1) // _BLOCK,)
        self._block = (_BLOCK,)

    # -- helpers ---------------------------------------------------------

    def _flat(self, array, name):
        import cupy as cp
        if not isinstance(array, cp.ndarray):
            raise TypeError(f"swint field {name!r} must be a device array")
        if array.dtype != np.float32 or array.shape != self.shape:
            raise ValueError(
                f"swint field {name!r} must be float32 {self.shape}, got "
                f"{array.dtype} {array.shape}")
        if not array.flags.c_contiguous:
            raise ValueError(f"swint field {name!r} must be C-contiguous")
        return array.reshape(-1)

    def coszen_loc_at(self, julian, xtime, gmt):
        """Fill and return the device ``coszen_loc`` (flat, ny*nx) for
        these calendar scalars (float32 julian, xtime in minutes, gmt in
        hours), at the latitude and longitude the carrier holds now."""
        if self.solar_generation == 'wrf_480':
            from woof.core.solar import solar_geometry_device
            _ds, cz, _ha = solar_geometry_device(julian, xtime, gmt,
                self.latitude_deg, self.longitude_deg, generation=self.solar_generation)
            self._coszen_loc[...] = cz.reshape(self.shape)
            return self._coszen_loc.reshape(-1)
        from woof.core.rrtmg_legacy import _DEGRAD, _DPD
        coszen_loc = self._coszen_loc.reshape(-1)
        self._k_coszen(self._grid, self._block,
                       (np.int64(self.ncol),
                        self._flat(self.latitude_deg, "latitude_deg"),
                        self._flat(self.longitude_deg, "longitude_deg"),
                        F(julian), F(xtime), F(gmt), F(_DEGRAD), F(_DPD), np.int32(self.solar_generation == "wrf_480"),
                        coszen_loc))
        return coszen_loc

    def coszen_loc(self, elapsed_seconds: float):
        """Fill and return the device ``coszen_loc`` at ``elapsed_seconds``."""
        return self.coszen_loc_at(
            *calendar_scalars(self.start_time, elapsed_seconds,
                              generation=self.solar_generation))

    def _interpolate(self, fields):
        f = {name: self._flat(fields[name], name) for name in STATE_FIELDS}
        # GSW takes the step's LIVE albedo.  v4.6.1 passes grid%albedo,
        # which the land surface rewrites between radiation calls (snow
        # cover, snow albedo); the fork-only sun-angle albedo (alb_sol = 1)
        # keeps it in ALBSOL, which the fork passes instead.  The albedo
        # captured at the last radiation call (swint_albedo, still carried
        # for restart) held GSW on that call's surface until the next
        # call, which neither code does.
        if "albsol" in fields:
            albedo = self._flat(fields["albsol"], "albsol")
        elif "albedo" in fields:
            albedo = self._flat(fields["albedo"], "albedo")
        else:
            raise KeyError(
                "swint_opt = 1 evaluates GSW = SWDOWN * (1 - ALBEDO) with "
                "the step's live albedo; fields carry neither 'albedo' nor "
                "'albsol'")
        self._k_interp(self._grid, self._block,
                       (np.int64(self.ncol), f["coszen_ref"],
                        self._coszen_loc.reshape(-1),
                        f["swddir_ref"], f["swint_bb"], f["swint_bx"],
                        f["swdown_ref"], f["swint_gg"], f["swint_gx"],
                        albedo, self._flat(fields["swdown"], "swdown"),
                        f["swddir"], f["swddni"], f["swddif"],
                        self._flat(fields["gsw"], "gsw")))

    # -- the two halves ----------------------------------------------------

    def radiation_step(self, fields, *, coszen, swddir, elapsed_seconds,
                       albedo=None):
        """The radiation-step half: fit, store the reference, evaluate.

        ``coszen`` is the radiation call's own (interval-midpoint) cosine
        and ``swddir`` the scheme's surface direct flux, both (ny, nx)
        device arrays; ``swddir=None`` means the scheme computes none and
        the driver's zeroed SWDDIR stands (:1091).  ``fields['swdown']``
        must already hold the call's SWDOWN and ``fields['albedo']`` the
        albedo the call ran with.
        """
        import cupy as cp
        f = {name: self._flat(fields[name], name) for name in STATE_FIELDS}
        coszen_loc = self.coszen_loc(elapsed_seconds)
        if swddir is None:
            swddir_rad = cp.zeros(self.ncol, dtype=cp.float32)
        else:
            swddir_rad = self._flat(swddir, "swddir")
        f["swint_albedo"][...] = self._flat(
            fields["albedo"] if albedo is None else albedo, "albedo")
        self._k_update(self._grid, self._block,
                       (np.int64(self.ncol), self._flat(coszen, "coszen"),
                        coszen_loc, swddir_rad,
                        self._flat(fields["swdown"], "swdown"),
                        f["swddir_ref"], f["swint_bb"], f["swint_bx"],
                        f["swdown_ref"], f["swint_gg"], f["swint_gx"],
                        f["coszen_ref"]))
        self._interpolate(fields)

    def between_calls(self, fields, *, elapsed_seconds):
        """The every-step half: evaluate the stored fit at the current sun."""
        self.coszen_loc(elapsed_seconds)
        self._interpolate(fields)
