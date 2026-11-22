"""WRF's slope-dependent shortwave and terrain shadowing on the device.

WRF v4.7.1 namelist ``&physics slope_rad`` (max_domains, default 0) turns
the shortwave the land surface receives into the flux on the local slope,
and ``topo_shading`` (max_domains, default 0) additionally removes the
direct beam where surrounding terrain hides the sun, searching up to
``shadlen`` metres (scalar, default 25000) toward it.  The pieces and where
WRF runs them:

* **static** -- SLOPE and SLP_AZI from HGT, once per domain
  (``dyn_em/start_em.F:1539-1577``): :func:`slope_geometry`.
* **each radiation call** -- the diffuse fraction of the surface shortwave
  (``module_radiation_driver.F:2894-2927``; RRTMG hands its own surface
  diffuse flux, a scheme without a direct beam gets the Ruiz-Arias split):
  :func:`diffuse_fraction`; and, with ``topo_shading = 1``, the shadow mask
  (``pre_radiation_driver`` -> ``toposhad_init`` + ``toposhad``,
  :4474-4862): :func:`terrain_shadow`.
* **each surface call** -- SWDOWN and GSW scaled for slope and shadow before
  the land surface (``module_surface_driver.F:1920-1936`` ->
  ``TOPO_RAD_ADJ_DRVR``, :6977-7114) and put back after it (:4461-4481),
  so the history keeps the flat flux and SWNORM carries the slope flux:
  :func:`adjust_surface_shortwave` / :func:`restore_surface_shortwave`.

WRF's own conditions carry over unchanged: the adjustment runs only with
``slope_rad = 1`` and a longwave scheme on (``radiation = ra_lw_physics >
0``, surface_driver:1918), and the shadow mask is read only by the
adjustment, so ``topo_shading = 1`` without ``slope_rad = 1`` computes
nothing WRF would use.

One patch covers the domain, as a serial wrf.exe runs it.  The kernels are
held bit for bit to WRF's own Fortran, linked to glibc 2.43's libm, by the
oracles in ``tools/wrf_topo_radiation_v471_oracle`` (each stage, WRF
v4.7.1) and ``tools/wrf_topo_radiation_v461_oracle`` (the whole chain over
realistic regimes, WRF v4.6.1), both graded in
``tests/test_topo_radiation.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_BLOCK = 256


def _kernel(name: str):
    from woof.core.kernels import get_kernel
    return get_kernel("topo_radiation", name)


def _grid(n: int) -> tuple[tuple[int], tuple[int]]:
    return ((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,)


def _f32(xp, array, shape, name):
    out = xp.ascontiguousarray(xp.asarray(array, dtype=xp.float32))
    if out.shape != shape:
        raise ValueError(f"{name} has shape {out.shape}, expected {shape}")
    return out


def slope_geometry(ht, msftx, msfty, sina, cosa, *, dx: float, dy: float,
                   initialization="column_oracle"):
    """``(slope, slp_azi)`` in radians from terrain height (start_em.F).

    ``rdx``/``rdy`` are WRF's ``1./dx`` and ``1./dy`` in single precision
    (the grid's REAL reciprocals).  The domain is not periodic.
    """
    import cupy as cp

    ht = cp.asarray(ht)
    ny, nx = ht.shape
    shape = (ny, nx)
    args = [_f32(cp, a, shape, n) for a, n in (
        (ht, "ht"), (msftx, "msftx"), (msfty, "msfty"), (sina, "sina"),
        (cosa, "cosa"))]
    slope = cp.empty(shape, dtype=cp.float32)
    slp_azi = cp.empty(shape, dtype=cp.float32)
    rdx = np.float32(1.0) / np.float32(dx)
    rdy = np.float32(1.0) / np.float32(dy)
    if initialization == "scalar_build":
        from woof.core.kernels import get_kernel_int_defines
        from woof.static.rust_bridge import scalar_rotation_asin_f32
        rotation = cp.asarray(scalar_rotation_asin_f32(cp.asnumpy(args[3])))
        kernel = get_kernel_int_defines("topo_radiation","topo_slope_geometry",(("TR_SLOPE_SCALAR",1),))
    elif initialization == "column_oracle":
        rotation = args[3]
        kernel = _kernel("topo_slope_geometry")
    else:
        raise ValueError("slope initialization must be column_oracle or scalar_build")
    kernel(
        *_grid(nx * ny), (*args, rotation, rdx, rdy, np.int32(nx), np.int32(ny),
                          slope, slp_azi))
    return slope, slp_azi


def widened_height(ht):
    """HGT with WRF's two-cell specified/nested halo on every side.

    ``set_physical_bc2d`` copies the edge mass value outward for a mass
    field (share/module_bc.F, the open_x*/open_y* copies, corners from the
    y pass over the whole memory row), so the halo is the nearest edge
    value.  toposhad's first-iteration scan on one patch reads two cells
    into it.
    """
    import cupy as cp

    ht = _f32(cp, ht, tuple(cp.asarray(ht).shape), "ht")
    return cp.ascontiguousarray(cp.pad(ht, 2, mode="edge"))


def terrain_shadow(ht, xlat, xlong, sina, cosa, *, xtime_minutes: float,
                   gmt: float, radt_minutes: float, declin: float,
                   dx: float, dy: float, shadlen: float,
                   parent_shadow=None):
    """``(shadowmask, ht_shad)`` for one radiation call (toposhad).

    ``xtime_minutes`` is WRF's XTIME at the call, ``radt_minutes`` its
    RADT (the scan's hour angle is the interval midpoint), ``declin`` the
    call's solar declination in radians.  ``parent_shadow`` is a nest's
    HT_SHAD as forced from its parent: only its two outer rows on each side
    are read (spec_bdyfield with spec_zone 2), and it switches on WRF's
    nested branch of toposhad_init; ``None`` is a root domain.
    """
    import cupy as cp

    ht = cp.asarray(ht)
    ny, nx = ht.shape
    shape = (ny, nx)
    ht_loc = widened_height(ht)
    nested = parent_shadow is not None
    ht_shad = (_f32(cp, parent_shadow, shape, "parent_shadow").copy()
               if nested else cp.zeros(shape, dtype=cp.float32))
    mask = cp.zeros(shape, dtype=cp.int32)
    _kernel("topo_shadow_init")(
        *_grid(nx * ny), (ht_loc, ht_shad, mask, np.int32(int(nested)),
                          np.int32(nx), np.int32(ny)))
    fields = [_f32(cp, a, shape, n) for a, n in (
        (xlat, "xlat"), (xlong, "xlong"), (sina, "sina"), (cosa, "cosa"))]
    _kernel("topo_shadow_scan")(
        *_grid(nx * ny), (ht_loc, *fields, np.float32(xtime_minutes),
                          np.float32(gmt), np.float32(radt_minutes),
                          np.float32(declin), np.float32(dx), np.float32(dy),
                          np.float32(shadlen), np.int32(nx), np.int32(ny),
                          mask, ht_shad))
    return mask, ht_shad


def diffuse_fraction(coszen, swdown, ht, swddif, *, solcon: float,
                     scheme_splits_direct_beam: bool):
    """DIFFUSE_FRAC for one radiation call; ``swddif`` updated in place.

    ``scheme_splits_direct_beam`` is True for RRTMG (its SWDDIF is the
    surface diffuse flux, SWDFLX - SWDKDIR); a shortwave scheme without its
    own direct beam gets WRF's Ruiz-Arias split first, into a zeroed
    ``swddif`` (the radiation driver zeroes it at the top of every call).
    """
    import cupy as cp

    coszen = cp.asarray(coszen)
    shape = tuple(coszen.shape)
    n = int(np.prod(shape))
    cz = _f32(cp, coszen, shape, "coszen")
    sw = _f32(cp, swdown, shape, "swdown")
    hh = _f32(cp, ht, shape, "ht")
    if not (isinstance(swddif, cp.ndarray) and swddif.dtype == cp.float32
            and swddif.flags.c_contiguous and swddif.shape == shape):
        raise ValueError("swddif must be a C-contiguous float32 device "
                         f"array of shape {shape}; it is updated in place")
    frac = cp.empty(shape, dtype=cp.float32)
    _kernel("topo_diffuse_frac")(
        *_grid(n), (cz, sw, hh, swddif, np.float32(solcon),
                    np.int32(0 if scheme_splits_direct_beam else 1),
                    np.int32(n), frac))
    return frac


@dataclass
class SurfaceShortwaveSave:
    """What TOPO_RAD_ADJ_DRVR saved, for the restore after the land call."""

    swnorm: object
    gswsave: object
    #: The GSW array the adjustment scaled (a scratch copy when the land
    #: surface reads SWDOWN only).
    gsw: object = None


def adjust_surface_shortwave(swdown, gsw, *, xlat, coszen, shadowmask,
                             diffuse_frac, hrang, slope, slp_azi,
                             declin: float) -> SurfaceShortwaveSave:
    """Scale SWDOWN and GSW in place for slope and shadow (TOPO_RAD_ADJ)."""
    import cupy as cp

    shape = tuple(swdown.shape)
    n = int(np.prod(shape))
    for name, array in (("swdown", swdown), ("gsw", gsw)):
        if not (isinstance(array, cp.ndarray) and array.dtype == cp.float32
                and array.flags.c_contiguous and array.shape == shape):
            raise ValueError(f"{name} must be a C-contiguous float32 device "
                             "array; it is adjusted in place")
    swnorm = cp.empty(shape, dtype=cp.float32)
    gswsave = cp.zeros(shape, dtype=cp.float32)
    mask = cp.ascontiguousarray(cp.asarray(shadowmask, dtype=cp.int32))
    if mask.shape != shape:
        raise ValueError("shadowmask shape differs from SWDOWN")
    _kernel("topo_rad_adjust")(
        *_grid(n), (_f32(cp, xlat, shape, "xlat"),
                    _f32(cp, coszen, shape, "coszen"), mask,
                    _f32(cp, diffuse_frac, shape, "diffuse_frac"),
                    _f32(cp, hrang, shape, "hrang"),
                    _f32(cp, slope, shape, "slope"),
                    _f32(cp, slp_azi, shape, "slp_azi"),
                    np.float32(declin), np.int32(n),
                    swdown, gsw, swnorm, gswsave))
    return SurfaceShortwaveSave(swnorm=swnorm, gswsave=gswsave)


def restore_surface_shortwave(swdown, gsw, save: SurfaceShortwaveSave):
    """Put the flat SWDOWN/GSW back; returns SWNORM (the slope flux)."""
    shape = tuple(swdown.shape)
    n = int(np.prod(shape))
    _kernel("topo_rad_restore")(
        *_grid(n), (swdown, gsw, save.swnorm, save.gswsave, np.int32(n)))
    return save.swnorm


# ---------------------------------------------------------------------------
# The per-domain driver object
# ---------------------------------------------------------------------------

def topo_shortwave_active(cfg) -> bool:
    """Whether WRF would adjust this domain's surface shortwave.

    ``slope_rad = 1`` with a longwave scheme on (surface_driver:1918 sets
    ``radiation = ra_lw_physics > 0``; a shortwave-only pair is WRF's own
    fatal) and a land-surface scheme to hand the flux to.  Anything else is
    WRF's no-op: the adjustment never runs, and topo_shading's mask has no
    reader.
    """
    from woof.config import radiation_scheme_ids

    lw, sw = radiation_scheme_ids(cfg)
    return (int(cfg.slope_rad) == 1 and lw > 0 and sw > 0
            and int(cfg.sf_surface_physics) > 0)


def request_surface_diffuse(radiation) -> None:
    """Ask a radiation callable (and its shortwave leaf) for SWDDIF."""
    for target in (radiation, getattr(radiation, "shortwave_adapter", None)):
        if target is not None:
            target.surface_diffuse_requested = True


class TopoShortwave:
    """One domain's slope_rad / topo_shading state and its three seams.

    Every array it holds lives in the physics driver's ``fields``, so a
    checkpoint carries it and a resume continues it bit for bit (the
    restart layer serialises the whole field set):

    * ``slope``, ``slp_azi``: static, from HGT at construction;
    * ``diffuse_frac``, ``topo_coszen``, ``hrang``, ``topo_declin``: held
      from the last radiation call (grid%diffuse_frac, coszen, hrang and
      declin in WRF), exactly as SWDOWN is held;
    * ``shadowmask``, ``ht_shad``: the last radiation call's shadow
      (topo_shading = 1 only).  On a nest, the outer two rows of
      ``ht_shad`` are what WRF feeds back into the next call: HT_SHAD is a
      forced field whose boundary VALUE table holds the nest's own current
      value (interp_fcn.F bdy_interp1 :2583-2584), and spec_bdyfield
      writes that value back before toposhad_init reads it;
    * ``swnorm``: WRF's SWNORM output, the slope-affected SWDOWN.
    """

    def __init__(self, *, state, cfg, fields, latitude, longitude,
                 start_time):
        import cupy as cp

        from woof.config import radiation_scheme_ids
        from woof.wrf_exact import ENABLED as strict_arithmetic

        ny, nx = tuple(state.ht.shape)
        self.shape = (ny, nx)
        self.start_time = start_time
        self.nested = bool(cfg.nested)
        self.shading = int(cfg.topo_shading) == 1
        self.dx, self.dy = float(cfg.dx), float(cfg.dy)
        self.shadlen = float(cfg.shadlen)
        self.scheme_splits_direct_beam = radiation_scheme_ids(cfg)[1] == 4
        self.xlat = _f32(cp, latitude, self.shape, "latitude")
        self.xlong = _f32(cp, longitude, self.shape, "longitude")
        self._xlat_host = cp.asnumpy(self.xlat).reshape(-1)
        self._xlong_host = cp.asnumpy(self.xlong).reshape(-1)
        slope, slp_azi = slope_geometry(
            state.ht, state.msft, state.msft, state.sina, state.cosa,
            dx=self.dx, dy=self.dy,
            initialization="scalar_build" if strict_arithmetic else "column_oracle")
        fields["slope"] = slope
        fields["slp_azi"] = slp_azi
        for name in ("diffuse_frac", "topo_coszen", "hrang", "swnorm"):
            fields[name] = cp.zeros(self.shape, dtype=cp.float32)
        fields["topo_declin"] = cp.zeros((1,), dtype=cp.float32)
        if self.shading:
            fields["shadowmask"] = cp.zeros(self.shape, dtype=cp.int32)
            fields["ht_shad"] = cp.zeros(self.shape, dtype=cp.float32)
        self.fields = fields

    # -- radiation seam ----------------------------------------------------
    def after_radiation(self, result, state, cfg) -> None:
        """What WRF's radiation driver leaves for the surface driver."""
        from datetime import timedelta

        import cupy as cp

        from woof.config import effective_radt_minutes

        F = np.float32
        f = self.fields
        # The WRF clock the radiation driver reads (as woof.core.
        # rrtmg_legacy transcribes it): fractional julian at the call, GMT
        # the start hour, XTIME the elapsed minutes, the hour angle at the
        # interval midpoint (module_radiation_driver.F:1206-1208).
        valid = self.start_time + timedelta(
            seconds=float(state.elapsed_seconds))
        hour = (valid.hour + valid.minute / 60.0 + valid.second / 3600.0
                + valid.microsecond / 3.6e9)
        julian = F((valid.timetuple().tm_yday - 1) + hour / 24.0)
        gmt = F(self.start_time.hour + self.start_time.minute / 60.0
                + self.start_time.second / 3600.0
                + self.start_time.microsecond / 3.6e9)
        xtime = F(float(state.elapsed_seconds) / 60.0)
        radt = F(effective_radt_minutes(cfg))
        generation = getattr(cfg, "solar_generation", "wrf_480")
        if generation == "wrf_480":
            from woof.core.solar import radiation_calendar
            julian, xtime, gmt = radiation_calendar(
                self.start_time, float(state.elapsed_seconds))
        from woof.wrf_exact import ENABLED as _strict
        if _strict and generation == "wrf_461":
            # The same rounded solar words consumed by strict legacy
            # radiation (rrtmg_legacy.solar_geometry_device); NumPy trig moved
            # the terrain cosine before the land-surface call.
            from woof.core.rrtmg_legacy import (
                solar_geometry_device as legacy_solar_geometry_device)
            coszen, hrang, declin, solcon = legacy_solar_geometry_device(
                julian, xtime + radt * F(0.5), gmt, self.xlat, self.xlong)
            f["topo_coszen"][...] = coszen.reshape(self.shape)
            f["hrang"][...] = hrang.reshape(self.shape)
        elif generation == "wrf_480":
            from woof.core.solar import solar_geometry_device
            ds, coszen, hrang = solar_geometry_device(julian, xtime + radt * F(0.5), gmt, self._xlat_host, self._xlong_host)
            declin, solcon = map(F, cp.asnumpy(ds))
            f["topo_coszen"][...] = cp.asarray(coszen.reshape(self.shape))
            f["hrang"][...] = cp.asarray(hrang.reshape(self.shape))
        else:
            from woof.core.rrtmg_legacy import calc_coszen_hrang, radconst
            declin, solcon = radconst(julian)
            coszen, hrang = calc_coszen_hrang(
                julian, xtime + radt * F(0.5), gmt, self._xlat_host,
                self._xlong_host, declin, generation='wrf_461')
            f["topo_coszen"][...] = cp.asarray(coszen.reshape(self.shape))
            f["hrang"][...] = cp.asarray(hrang.reshape(self.shape))
        f["topo_declin"][...] = F(declin)
        if self.scheme_splits_direct_beam:
            if result.swddif is None:
                raise ValueError(
                    "slope_rad = 1 needs the RRTMG surface diffuse flux "
                    "(SWDDIF) from the radiation call, and the attached "
                    "scheme returned none")
            swddif = cp.ascontiguousarray(
                cp.asarray(result.swddif, dtype=cp.float32))
        else:
            # Zeroed at the top of every WRF radiation call (:1726); the
            # Ruiz-Arias split fills the sunlit columns.
            swddif = cp.zeros(self.shape, dtype=cp.float32)
        f["diffuse_frac"][...] = diffuse_fraction(
            f["topo_coszen"], f["swdown"], state.ht, swddif,
            solcon=float(solcon),
            scheme_splits_direct_beam=self.scheme_splits_direct_beam)
        if self.shading:
            mask, shad = terrain_shadow(
                state.ht, self.xlat, self.xlong, state.sina, state.cosa,
                xtime_minutes=float(xtime), gmt=float(gmt),
                radt_minutes=float(radt), declin=float(declin),
                dx=self.dx, dy=self.dy, shadlen=self.shadlen,
                parent_shadow=f["ht_shad"] if self.nested else None)
            f["shadowmask"][...] = mask
            f["ht_shad"][...] = shad

    # -- surface seam ------------------------------------------------------
    def before_land(self):
        """TOPO_RAD_ADJ_DRVR, before the land surface reads the flux."""
        import cupy as cp

        f = self.fields
        mask = (f["shadowmask"] if self.shading
                else cp.zeros(self.shape, dtype=cp.int32))
        # GSW is a driver field only under RUC; Noah and Noah-MP read
        # SWDOWN.  WRF scales both, and a GSW nobody reads is scaled into
        # a scratch copy and dropped, as WRF's restore drops it.
        gsw = f["gsw"] if "gsw" in f else cp.zeros_like(f["swdown"])
        save = adjust_surface_shortwave(
            f["swdown"], gsw, xlat=self.xlat, coszen=f["topo_coszen"],
            shadowmask=mask, diffuse_frac=f["diffuse_frac"],
            hrang=f["hrang"], slope=f["slope"], slp_azi=f["slp_azi"],
            declin=float(cp.asnumpy(f["topo_declin"])[0]))
        save.gsw = gsw
        return save

    def after_land(self, save) -> None:
        """Put the flat SWDOWN/GSW back; SWNORM keeps the slope flux."""
        self.fields["swnorm"][...] = restore_surface_shortwave(
            self.fields["swdown"], save.gsw, save)
