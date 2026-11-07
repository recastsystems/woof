"""Goddard 4-ice microphysics with hail (NU-WRF), WRF ``mp_physics = 7``.

WRF v4.6.1 ``phys/module_mp_gsfcgce_4ice_nuwrf.F`` on the device:
``kernels/gsfcgce4ice.cu`` carries consat_s, fall_flux, saticel_s and their
helpers, checked by the column oracle against the scheme object WRF's own build
produced (``tools/gsfcgce4ice_wrf461_oracle``).

What this module owns is the launch: the constant table consat_s builds
(once per device, through the immutable-table cache every uploaded table
uses), WRF's microphysics prep for the call, and the outputs the scheme
writes that no other scheme has -- hail precipitation, the six latent
heating rates and their accumulators, the six RE_*_GSFC radii and the five
sedimentation fluxes.  What the scheme IS lives in the registry row
(``woof/physics_registry_v2.json``, option ``goddard4ice-mp7``).
"""

from __future__ import annotations

import numpy as np

from woof.core.device_cache import cuda_cache

#: The kernel's G4_TABLE_WORDS: consat_s's module constants plus the
#: per-call constants saticel_s derives from them without dt.
TABLE_WORDS = 512
_COLUMN_TPB = 64

#: The per-level fields the scheme writes beside the moist species, in the
#: kernel's argument order.  Names are woof state attributes; the WRF
#: Registry names are the upper-case spellings (PHYSC, ACPHYSC,
#: RE_CLOUD_GSFC, PRECR3D, ...).
HEATING_FIELDS = ("physc", "physe", "physd", "physs", "physm", "physf")
ACCUMULATED_HEATING_FIELDS = ("acphysc", "acphyse", "acphysd", "acphyss",
                              "acphysm", "acphysf")
RADIUS_FIELDS = ("re_cloud_gsfc", "re_rain_gsfc", "re_ice_gsfc",
                 "re_snow_gsfc", "re_graupel_gsfc", "re_hail_gsfc")
FLUX_FIELDS = ("preci3d", "precs3d", "precg3d", "prech3d", "precr3d")
#: Every per-level field mp_physics=7 adds to a moist state beyond
#: qi/qs/qg/qh.
SCHEME_LEVEL_FIELDS = (HEATING_FIELDS + ACCUMULATED_HEATING_FIELDS
                       + RADIUS_FIELDS + FLUX_FIELDS)
SURFACE_SLOTS = ("mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
                 "mp_sr", "mp_graupelnc", "mp_graupelncv", "mp_hailnc",
                 "mp_hailncv")


@cuda_cache(maxsize=None, ready=True)
def constant_table():
    """consat_s, run once per device into an immutable table."""
    import cupy as cp
    from woof.core.kernels import get_kernel

    table = cp.zeros(TABLE_WORDS, dtype=cp.float32)
    get_kernel("gsfcgce4ice", "gsfc4ice_consat")((1,), (1,), (table,))
    return table


def launch_gsfcgce4ice(*, th, qv, qc, qr, qi, qs, qg, qh, rho, pii, p, z,
                       dz8w, w, ht, xland, refl_10cm, heating, accumulated,
                       radii, fluxes, rainnc, rainncv, snownc, snowncv, sr,
                       graupelnc, graupelncv, hailnc, hailncv,
                       dt: float, dx: float, itimestep: int,
                       refl_due: bool, vt=None, err=None, module=None):
    """One gsfcgce_4ice_nuwrf call over contiguous float32 fields.

    Level fields are ``(nz, ny, nx)``; ``w`` may carry the staggered
    ``nz + 1`` levels (only the lowest ``nz`` are read, as WRF reads w at
    each mass level's lower face); surface fields are ``(ny, nx)``.
    ``heating``/``accumulated``/``radii``/``fluxes`` are sequences in the
    order of :data:`HEATING_FIELDS` and friends.  ``module`` lets a test
    supply an alternative compiled translation unit (the planted-change
    check); production passes ``None``.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    dtype = np.dtype(np.float32)
    shape = th.shape
    if len(shape) != 3:
        raise ValueError(f"Goddard 4-ice fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"dx must be finite and positive, got {dx}")
    volume = {"th": th, "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs,
              "qg": qg, "qh": qh, "rho": rho, "pii": pii, "p": p, "z": z,
              "dz8w": dz8w, "refl_10cm": refl_10cm}
    for i, a in enumerate(tuple(heating) + tuple(accumulated) + tuple(radii)
                          + tuple(fluxes)):
        volume[f"scheme_field_{i}"] = a
    if (len(heating), len(accumulated), len(radii), len(fluxes)) != (6, 6, 6, 5):
        raise ValueError("heating/accumulated/radii/fluxes must be 6/6/6/5 "
                         "fields")
    for name, value in volume.items():
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"{name} must be float32 {shape}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    if (w.shape not in ((nz, ny, nx), (nz + 1, ny, nx)) or w.dtype != dtype
            or not w.flags.c_contiguous):
        raise ValueError(f"w must be C-contiguous float32 (nz[+1], ny, nx), "
                         f"got {w.dtype} {w.shape}")
    surface = {"ht": ht, "xland": xland, "rainnc": rainnc,
               "rainncv": rainncv, "snownc": snownc, "snowncv": snowncv,
               "sr": sr, "graupelnc": graupelnc, "graupelncv": graupelncv,
               "hailnc": hailnc, "hailncv": hailncv}
    for name, value in surface.items():
        if value.shape != (ny, nx) or value.dtype != dtype:
            raise ValueError(f"{name} must be float32 {(ny, nx)}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    ncol = ny * nx
    if vt is None:
        vt = cp.empty(shape, dtype=cp.float32)
    if err is None:
        err = cp.empty(1, dtype=cp.float32)
    if err.shape != (1,) or err.dtype != dtype or not err.flags.c_contiguous:
        raise ValueError("err must be C-contiguous float32 (1,)")
    err.fill(0)
    kernel = (get_kernel("gsfcgce4ice", "gsfc4ice_column") if module is None
              else module.get_function("gsfc4ice_column"))
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel((blocks,), (_COLUMN_TPB,), (
        constant_table() if module is None else _table_for(module),
        th, qv, qc, qr, qi, qs, qg, qh, rho, pii, p, z, dz8w, w, ht, xland,
        refl_10cm, *heating, *accumulated, *radii, *fluxes,
        rainnc, rainncv, snownc, snowncv, sr, graupelnc, graupelncv,
        hailnc, hailncv, vt, err,
        np.float32(dt), np.float32(dx), np.int32(itimestep),
        np.int32(1 if refl_due else 0), np.int32(nz), np.int32(ncol)))
    return err


def _table_for(module):
    import cupy as cp
    table = cp.zeros(TABLE_WORDS, dtype=cp.float32)
    module.get_function("gsfc4ice_consat")((1,), (1,), (table,))
    return table


def refuse_unknown_land_mask(xland) -> None:
    """XLAND must be 1 (land) or 2 (water) in every column.

    The breakage this prevents: module_mp_gsfcgce_4ice_nuwrf.F:4412-4420
    picks the cloud-droplet number for the cloud effective radius from
    XLAND and stops the run (wrf_error_fatal) for any other value; a
    fabricated number would hand radiation a radius WRF never computes.
    """
    import cupy as cp
    bad = int(cp.count_nonzero((xland != np.float32(1.0))
                               & (xland != np.float32(2.0))))
    if bad:
        raise ValueError(
            f"mp_physics=7 (Goddard 4-ice) needs XLAND of exactly 1 or 2 in "
            f"every column for its cloud effective radius (module_mp_"
            f"gsfcgce_4ice_nuwrf.F:4412-4420, where WRF stops the run); "
            f"{bad} column(s) carry another value")


__all__ = ["constant_table", "launch_gsfcgce4ice", "refuse_unknown_land_mask",
           "HEATING_FIELDS", "ACCUMULATED_HEATING_FIELDS", "RADIUS_FIELDS",
           "FLUX_FIELDS", "SCHEME_LEVEL_FIELDS", "SURFACE_SLOTS",
           "TABLE_WORDS"]


def _aux_module(nz=40):
    from woof.core.kernels import load_module_int_defines
    if not 1 <= nz <= 256:
        raise ValueError('The standalone reflectivity array bound is 1..256 levels; a larger column would overrun its device workspace.')
    return load_module_int_defines('gsfcgce4ice_aux', (('G4_AUX_KMAX', int(nz)),))


def _aux_fields(*fields):
    shape = fields[0].shape
    for value in fields:
        if value.shape != shape or value.dtype != np.float32 or not value.flags.c_contiguous:
            raise ValueError('The helper fields must share one contiguous float32 layout; mismatched geometry would read another field or overrun its storage.')
    return shape


@cuda_cache(maxsize=None, ready=True)
def radar_table():
    """WRF radar_init products, built once on the device without table files.

    The active entry's incoming water and snow densities are 1000 and
    100 kg/m3. Its graupel density and intercepts come from consat_s.
    Fixed optical coefficients at the source's zero-temperature radar
    wavelength are compile-time values pinned by the Fortran init oracle.
    """
    import cupy as cp
    table = cp.zeros(269, dtype=cp.float64)
    _aux_module().get_function('g4_aux_init')((1,), (1,), (table, constant_table()))
    return table


def launch_negcor(x, rho, dz, *, i_start=0, i_end=None, j_start=0, j_end=None,
                  k_start=0, k_end=None, module=None):
    """WRF's serial tile-wide mass correction, including its no-correction arm.

    This preserves the k,j,i reduction order. WRF disables this call in
    the active mp7 entry, so this explicit helper does not alter a forecast.
    Tile endpoints are half-open Python indexes. WRF's unit-61 debug
    prints are not emitted; all numerical writes are preserved.
    """
    shape = _aux_fields(x, rho, dz)
    if len(shape) != 3:
        raise ValueError('negcor requires (level,y,x) fields to preserve the WRF tile reduction order.')
    nz, ny, nx = shape
    ie = nx if i_end is None else i_end
    je = ny if j_end is None else j_end
    ke = nz if k_end is None else k_end
    for lo, hi, size in ((i_start, ie, nx), (j_start, je, ny), (k_start, ke, nz)):
        if not isinstance(lo, (int, np.integer)) or not isinstance(hi, (int, np.integer)):
            raise ValueError('negcor tile indexes must be integers; truncating a fraction would select different cells.')
        if not 0 <= lo <= hi <= size:
            raise ValueError('The negcor tile lies outside its arrays and would read or write unrelated device storage.')
    mod = _aux_module() if module is None else module
    args = (x, rho, dz) + tuple(np.int32(v) for v in (nx, ny, nz, i_start, ie, j_start, je, k_start, ke))
    mod.get_function('g4_aux_negcor')((1,), (1,), args)


def launch_auto_conversion(liquid, number, rate, radius, *, module=None):
    """WRF auto_conversion with its fixed no_thresh arm.

    Liquid is g/cm3, number is particles/cm3, rate is g/cm3/s and radius
    is micrometres. WRF leaves radius unwritten on an early return; the
    caller's existing radius word remains untouched in those elements.
    """
    _aux_fields(liquid, number, rate, radius)
    n = liquid.size
    if n == 0:
        return
    mod = _aux_module() if module is None else module
    mod.get_function('g4_aux_auto')(((n + 63) // 64,), (64,),
        (liquid, number, rate, radius, np.int32(n)))


def launch_refl10cm_gsfc(qv, qr, qs, qg, temperature, pressure, output, *, module=None):
    """The uncalled standalone WRF radar helper, without an added dBZ floor.

    This helper's species are rain, snow and graupel. The active mp7
    entry instead writes its own hail-inclusive saticel_s reflectivity.
    """
    shape = _aux_fields(qv, qr, qs, qg, temperature, pressure, output)
    if len(shape) != 3:
        raise ValueError('The radar helper needs (level,y,x) columns for its melting-level scan.')
    nz, ny, nx = shape
    mod = _aux_module(nz) if module is None else module
    ncol = nx * ny
    mod.get_function('g4_aux_refl')(((ncol + 31) // 32,), (32,),
        (qv, qr, qs, qg, temperature, pressure, output, radar_table(),
         np.int32(nx), np.int32(ny), np.int32(nz)))
