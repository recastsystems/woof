"""mp=28 water/ice-friendly (WIF) aerosol climatology ingest.

Port of WRF's ``QNWFA_QNIFA_SIGMA_MONTHLY.dat`` input path -- the GLOBAL
monthly Thompson aerosol-aware climatology -- with WRF as the numerical
authority at every stage:

1. **Decode**: the dataset ships in WPS intermediate format (ungrib IFV=5,
   big-endian Fortran sequential records; ``WPS/ungrib/src/read_met_module``).
   The copy on the reference host is 288x181 global cylindrical-equidistant
   (SWCORNER -90,-180, dlat 1.0, dlon 1.25 -- read from the file's own
   projection records, never assumed), 30 levels x 12 months x
   {QNWFA, QNIFA, P_WIF}.
2. **Horizontal**: metgrid maps every monthly level with
   ``interp_option=four_pt+average_4pt`` (``WPS/metgrid/METGRID.TBL:885-1150``).
   ``four_pt`` is four-point bilinear in fractional source-index space; on a
   fully valid global field the ``average_4pt`` fallback is unreachable, so
   bilinear IS the whole operator.
3. **Temporal**: real.exe's ``monthly_interp_to_date``
   (``dyn_em/module_initialize_real.F:8029-8095``): integer julian-day
   linear weighting between month middles (the 15th), hour-of-day ignored.
4. **Vertical**: real.exe's ``vert_interp`` climatology calls
   (``module_initialize_real.F:2452/:2519``): the temporally interpolated
   column (flipped bottom-up when the pressure field says so, :2367-2372)
   is assembled with the level-1 value in the surface slot, zap-close-levels
   and force_sfc_in_vinterp applied exactly as :6035-6086, then linear in
   LOG(p) (``linear_interp`` Registry default 1, ``interp_type`` default 2)
   onto the model's DRY half-level eta pressure -- the ``grid%pb`` argument
   of the WIF call is, at that point in init_domain_rk, a SCRATCH array
   holding ``p_dry(mu0, znw, p_top)`` on half levels
   (module_initialize_real.F:1625-1629, :1701), i.e. ``c3h(k)*mu0+c4h(k)+
   p_top`` -- the same target every met-field vert_interp call uses, NOT
   the final base-state pressure assigned at :3795.
   Below the deepest assembled point extrapolation is constant
   (``extrap_type`` default 2, var_type 'Q'); a target above the source top
   is a hard error, as in WRF.
5. **Surface emission**: ``qnwfa2d = w_wif_now(:,1,:) * 0.000196 * (50/z1)``
   with ``z1 = (phb(2)-phb(1))/g`` and ``qnifa2d = 0``
   (``module_initialize_real.F:4530-4547``).

The ARBITRARY-ACCEPTANCE argument, stated once so the design carries it:
this dataset does NOT come from the driving model.  WRF consumes it through
metgrid's ``constants_name`` mechanism, identically for every first-guess
source, and everything grid-specific here is metadata (the file's own
projection records, the model grid's lat/lon and base state).  Porting it
therefore serves every input source at once; there is no per-source branch
anywhere in this module.

ENGINE (2.5.0 data-path law): the data path is WOOF's Rust.  Decode,
horizontal interpolation and vertical interpolation run in the packaged
CPU bridge (``tools/grib1_bridge``, cdylib ``gpuwm_preprocess_cpu``) --
``gpuwm_wps_intermediate_read``, ``gpuwm_regular_cyclic_bilinear_f32``
and the pre-existing ``gpuwm_wrf_vert_interp_f32``, reached through
:class:`woof.ingest.cpu_backend.CpuPreprocessBackend`.

The NumPy implementations in this module are RETAINED, and they are not
a shipped fallback: they are the ORACLE-OF-RECORD for the equivalence
gate.  These are the exact expressions that were measured against
WRF-4.7.1 real.exe on test host D (commit 94260bf44: QNWFA maxabs
1.484800e+04 on a field maxing 4.003303e+09, QNIFA maxabs 1.781250e+00
on 5.978444e+05, QNWFA2D bit-exact from metgrid's own horizontal
output), so they are the only thing that can say the Rust engine still
holds those numbers.  ``backend="numpy"`` therefore exists for the gate
and for tests, and the shipped route -- the one ``real.py`` calls -- is
``backend="rust"``.  A run whose bridge is missing is REFUSED, not
silently demoted to NumPy: a demotion would move the data path off the
engine the receipt claims, which is the precise breakage the 2.5.0 law
names.

Two of the three Rust entries are NEW GENERIC OPERATORS, not a
WIF-shaped path.  ``met_intermediate`` in the same crate already WROTE
the WPS intermediate format and nothing could read it back; and the
existing regular bilinear clamps its last column, which is right for a
bounded source and wrong at a GLOBAL source's seam.  Both gaps belong
to every static dataset WPS routes through ``constants_name``, and both
are now closed for all of them.  The third entry is reused unchanged.
"""

from __future__ import annotations

import hashlib
import os
import struct
import threading
from dataclasses import dataclass, field as _field

import numpy as np

#: Gravity as WRF's share/module_model_constants.F g (real.exe uses it for
#: the phb -> z1 conversion feeding qnwfa2d).
_G = 9.81

#: real.exe namelist defaults that the WIF vert_interp call consumes
#: (Registry.EM_COMMON): interp_type=2 (log p), extrap_type=2 (constant for
#: 'Q'), linear_interp=1 (:2321), zap_close_levels=500 Pa,
#: force_sfc_in_vinterp=1.  The WIF call hard-codes lowest_lev_from_sfc
#: .false. and passes ``linear_interp`` in the lagrange-order slot
#: (module_initialize_real.F:2452-2465), so the operator is linear at every
#: eta level regardless of the met-field lagrange_order.
ZAP_CLOSE_LEVELS_PA = 500.0
FORCE_SFC_IN_VINTERP = 1

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

#: Cumulative days to month start, non-leap / leap -- for the same integer
#: julian-day arithmetic get_julgmt feeds monthly_interp_to_date.
_MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


class WifClimatologyError(ValueError):
    """A WIF climatology input violated the WRF-defined contract."""


@dataclass(frozen=True)
class WifClimatology:
    """The decoded global monthly WIF dataset, month-major, file level order.

    ``qnwfa``/``qnifa``/``pressure`` are ``(12, nlev, nlat, nlon)`` float32;
    ``latitude``/``longitude`` are the file's own 1-D axes (degrees).
    """

    qnwfa: np.ndarray
    qnifa: np.ndarray
    pressure: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    source_path: str
    #: The file's own SWCORNER and increments, kept verbatim so the
    #: horizontal operator rebuilds the axes from the same two numbers
    #: the decoder read rather than re-deriving them from a rounded
    #: axis.  ``None`` means "recover them from the axes", which is what
    #: a caller that built this object by hand gets.
    grid_corner: tuple | None = _field(default=None)

    def axis_parameters(self):
        """``(startlat, deltalat, startlon, deltalon)`` in FP64."""

        if self.grid_corner is not None:
            return tuple(float(value) for value in self.grid_corner)
        return (float(self.latitude[0]),
                float(self.latitude[1] - self.latitude[0]),
                float(self.longitude[0]),
                float(self.longitude[1] - self.longitude[0]))



#: The one place the engine choice is spelled.  "rust" is the shipped
#: data path (2.5.0 law); "numpy" is the oracle-of-record used by the
#: equivalence gate and the unit tests, never by a run.
WIF_BACKENDS = ("rust", "numpy")


def _resolve_backend(backend):
    """Refuse an unknown engine by name; never guess one."""

    if backend not in WIF_BACKENDS:
        raise WifClimatologyError(
            f"unknown WIF ingest backend {backend!r}; the data path is "
            f"one of {WIF_BACKENDS} -- 'rust' is the shipped engine and "
            "'numpy' is the equivalence oracle")
    return backend


def _cpu_backend():
    """The packaged Rust CPU bridge, or a refusal that names what is missing.

    A missing bridge is NOT demoted to the NumPy reference.  The receipt
    this module publishes names the Rust engine, and a silent demotion
    would make that receipt false while producing numbers that differ in
    the last bits -- which is exactly the data-path drift the 2.5.0 law
    exists to prevent.
    """

    from woof.ingest.cpu_backend import CpuPreprocessBackend

    backend = CpuPreprocessBackend()
    if not backend.wps_intermediate_reader or not backend.cyclic_bilinear:
        raise WifClimatologyError(
            f"the CPU preprocessing bridge at {backend.path} predates the "
            "static-dataset entries (gpuwm_wps_intermediate_read, "
            "gpuwm_regular_cyclic_bilinear_f32); rebuild "
            "tools/grib1_bridge. Refusing rather than running the NumPy "
            "reference, which is the equivalence oracle and not a "
            "shipped data path")
    return backend

def _read_record(handle):
    head = handle.read(4)
    if not head:
        return None
    (length,) = struct.unpack(">i", head)
    payload = handle.read(length)
    (tail,) = struct.unpack(">i", handle.read(4))
    if tail != length or len(payload) != length:
        raise WifClimatologyError(
            "corrupt Fortran sequential record in WIF climatology file")
    return payload


def read_wps_intermediate(path):
    """Read every field record of a WPS intermediate (IFV=5) file.

    Returns ``(records, latitude, longitude)`` where ``records`` maps
    ``field_name -> {xlvl: (nlat, nlon) float32}``.  Only ``iproj=0``
    (cylindrical equidistant) is admitted, because that is the projection
    the WIF climatology declares; any other projection in a file handed to
    this reader is refused by name rather than misinterpreted.
    """
    records: dict[str, dict[float, np.ndarray]] = {}
    axes = None
    with open(path, "rb") as handle:
        while True:
            rec = _read_record(handle)
            if rec is None:
                break
            if len(rec) != 4:
                raise WifClimatologyError(
                    "expected IFV version record in WPS intermediate file")
            (version,) = struct.unpack(">i", rec)
            if version != 5:
                raise WifClimatologyError(
                    f"unsupported WPS intermediate version {version}; the "
                    "reader ports the IFV=5 layout the WIF climatology uses")
            header = _read_record(handle)
            field = header[60:69].decode("ascii").strip()
            xlvl, nx, ny, iproj = struct.unpack(">fiii", header[140:156])
            proj = _read_record(handle)
            if iproj != 0:
                raise WifClimatologyError(
                    f"field {field!r} declares iproj={iproj}; only the "
                    "cylindrical-equidistant (iproj=0) layout of the WIF "
                    "climatology is ported")
            startlat, startlon, deltalat, deltalon, _earth_radius = (
                struct.unpack(">fffff", proj[8:28]))
            _is_wind_earth_rel = _read_record(handle)
            data = _read_record(handle)
            if len(data) != 4 * nx * ny:
                raise WifClimatologyError(
                    f"field {field!r} data record does not match nx*ny")
            values = np.frombuffer(data, dtype=">f4").astype(
                np.float32).reshape(ny, nx)
            this_axes = (
                np.float64(startlat) + np.float64(deltalat) * np.arange(ny),
                np.float64(startlon) + np.float64(deltalon) * np.arange(nx),
            )
            if axes is None:
                axes = this_axes
            elif (not np.array_equal(axes[0], this_axes[0])
                  or not np.array_equal(axes[1], this_axes[1])):
                raise WifClimatologyError(
                    "WIF climatology fields disagree on the source grid")
            records.setdefault(field, {})[float(xlvl)] = values
    if axes is None:
        raise WifClimatologyError("WPS intermediate file holds no records")
    return records, axes[0], axes[1]



def read_wps_intermediate_rust(path):
    """:func:`read_wps_intermediate`, decoded by the Rust bridge.

    Same return shape, same refusals.  The Rust side returns every
    record's header verbatim; the projection admission and the
    one-grid-per-file check are policy and stay here, stated once.
    """

    backend = _cpu_backend()
    metas, data = backend.read_wps_intermediate(path)
    records: dict[str, dict[float, np.ndarray]] = {}
    axes = None
    corner = None
    for meta in metas:
        field = meta["field"]
        if meta["iproj"] != 0:
            raise WifClimatologyError(
                f"field {field!r} declares iproj={meta['iproj']}; only the "
                "cylindrical-equidistant (iproj=0) layout of the WIF "
                "climatology is ported")
        ny, nx = meta["ny"], meta["nx"]
        start = meta["offset"]
        values = data[start:start + ny * nx].reshape(ny, nx)
        this_axes = (
            np.float64(meta["startlat"])
            + np.float64(meta["deltalat"]) * np.arange(ny),
            np.float64(meta["startlon"])
            + np.float64(meta["deltalon"]) * np.arange(nx),
        )
        if axes is None:
            axes = this_axes
            corner = (meta["startlat"], meta["deltalat"],
                      meta["startlon"], meta["deltalon"])
        elif (not np.array_equal(axes[0], this_axes[0])
              or not np.array_equal(axes[1], this_axes[1])):
            raise WifClimatologyError(
                "WIF climatology fields disagree on the source grid")
        records.setdefault(field, {})[float(meta["xlvl"])] = values
    if axes is None:
        raise WifClimatologyError("WPS intermediate file holds no records")
    return records, axes[0], axes[1], corner

#: Decoded climatologies and their horizontal interpolations, per process.
#: THE BREAKAGE: a GFS preparation decoded the same monthly WIF file and
#: interpolated all 12 months of it onto the same grid once PER FORCING
#: TIME (13 times for a 36 h 3-hourly window: about 12 s of a 50 s
#: preparation, measured 2026-10-27).  The file and the grid do not
#: change between forcing times; only the date weights and the vertical
#: interpolation do.  The cached arrays are read-only, so a consumer that
#: tried to write one fails loudly instead of changing a later time.
_DECODED: dict = {}
_HORIZONTAL: dict = {}
_CACHE_ENTRIES = 2


_CACHE_LOCK = threading.Lock()


def _remember(cache: dict, key, value):
    with _CACHE_LOCK:
        cache.pop(key, None)
        cache[key] = value
        while len(cache) > _CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
    return value


def _frozen(array):
    array = np.asarray(array)
    array.setflags(write=False)
    return array


def load_wif_climatology(path, *, backend="rust") -> WifClimatology:
    """Assemble the (12, nlev, nlat, nlon) monthly stacks by field suffix.

    ``backend`` selects the DECODER only; the stacking below is metadata
    work on whatever the decoder returned and is identical either way.

    Decoded once per process per file (path, size, mtime) and decoder
    (:data:`_DECODED`); the arrays returned are read-only.
    """

    engine = _resolve_backend(backend)
    try:
        stat = os.stat(path)
        key = (os.path.realpath(path), stat.st_size, stat.st_mtime_ns, engine)
    except (OSError, TypeError, ValueError):
        key = None
    if key is not None and key in _DECODED:
        return _remember(_DECODED, key, _DECODED[key])
    decoded = _decode_wif_climatology(path, engine)
    climatology = WifClimatology(
        qnwfa=_frozen(decoded.qnwfa), qnifa=_frozen(decoded.qnifa),
        pressure=_frozen(decoded.pressure),
        latitude=_frozen(decoded.latitude),
        longitude=_frozen(decoded.longitude),
        source_path=decoded.source_path, grid_corner=decoded.grid_corner)
    if key is not None:
        _remember(_DECODED, key, climatology)
    return climatology


def _decode_wif_climatology(path, engine) -> WifClimatology:
    if engine == "rust":
        records, latitude, longitude, corner = read_wps_intermediate_rust(path)
    else:
        records, latitude, longitude = read_wps_intermediate(path)
        corner = None

    def stack(prefix):
        months = []
        for month in _MONTHS:
            name = f"{prefix}_{month}"
            if name not in records:
                raise WifClimatologyError(
                    f"WIF climatology is missing {name}; real.exe treats a "
                    "missing month as fatal "
                    "(module_initialize_real.F:2410-2412) and so does this "
                    "port")
            by_level = records[name]
            levels = sorted(by_level)
            months.append(np.stack([by_level[lvl] for lvl in levels]))
        stacked = np.stack(months)
        if stacked.shape != months[0].shape[:0] + (12,) + months[0].shape:
            raise WifClimatologyError("inconsistent WIF level counts")
        return stacked

    qnwfa = stack("QNWFA")
    qnifa = stack("QNIFA")
    pressure = stack("P_WIF")
    if not (qnwfa.shape == qnifa.shape == pressure.shape):
        raise WifClimatologyError(
            "QNWFA/QNIFA/P_WIF level counts disagree in the WIF climatology")
    return WifClimatology(qnwfa=qnwfa, qnifa=qnifa, pressure=pressure,
                          latitude=latitude, longitude=longitude,
                          source_path=str(path), grid_corner=corner)


def four_pt_bilinear(field, latitude, longitude, target_lat, target_lon):
    """metgrid ``four_pt`` on a regular (optionally global) lat-lon source.

    ``field`` is ``(..., nlat, nlon)``; the last two axes are interpolated
    to the flattened target points and reshaped to ``field.shape[:-2] +
    target_lat.shape``.  Longitude wraps modulo 360 so a global source has
    no seam; latitude poleward of the axis is refused (the WIF grid spans
    the full -90..90, so a real target never trips it).
    """
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    dlat = latitude[1] - latitude[0]
    dlon = longitude[1] - longitude[0]
    nlat, nlon = latitude.size, longitude.size
    span = nlon * dlon
    global_lon = abs(abs(span) - 360.0) < 1.0e-6

    y = (target_lat.ravel() - latitude[0]) / dlat
    x = (target_lon.ravel() - longitude[0]) / dlon
    if global_lon:
        x = np.mod(x, nlon)
    if (y.min() < 0.0) or (y.max() > nlat - 1):
        raise WifClimatologyError(
            "target latitude escapes the WIF climatology grid")
    if not global_lon and ((x.min() < 0.0) or (x.max() > nlon - 1)):
        raise WifClimatologyError(
            "target longitude escapes the non-global source grid")

    y0 = np.clip(np.floor(y).astype(np.int64), 0, nlat - 2)
    x0 = np.floor(x).astype(np.int64)
    fy = (y - y0).astype(np.float64)
    fx = (x - x0).astype(np.float64)
    x1 = x0 + 1
    if global_lon:
        x0 = np.mod(x0, nlon)
        x1 = np.mod(x1, nlon)
    w00 = (1.0 - fx) * (1.0 - fy)
    w01 = fx * (1.0 - fy)
    w10 = (1.0 - fx) * fy
    w11 = fx * fy
    field = np.asarray(field)
    flat = field.reshape(field.shape[:-2] + (nlat * nlon,))
    idx00 = y0 * nlon + x0
    idx01 = y0 * nlon + x1
    idx10 = (y0 + 1) * nlon + x0
    idx11 = (y0 + 1) * nlon + x1
    out = (flat[..., idx00] * w00 + flat[..., idx01] * w01
           + flat[..., idx10] * w10 + flat[..., idx11] * w11)
    return out.reshape(field.shape[:-2] + target_lat.shape).astype(np.float32)


def monthly_interp_weights(date_str):
    """``monthly_interp_to_date``'s integer julian bracketing and weights.

    Returns ``(month1, month2, weight1_days, weight2_days, denominator)``
    with months 1-based, exactly the integer arithmetic of
    ``module_initialize_real.F:8059-8092`` (year*1000+julday resolution:
    the hour of day never enters).
    """
    year = int(date_str[0:4])
    month = int(date_str[5:7])
    day = int(date_str[8:10])
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def julday(m, d):
        extra = 1 if (leap and m > 2) else 0
        return _MONTH_START[m - 1] + d + extra

    middle = [0] * 14
    for m in range(1, 13):
        middle[m] = year * 1000 + julday(m, 15)
    middle[0] = middle[1] - 31
    middle[13] = middle[12] + 31
    target = year * 1000 + julday(month, day)
    for l in range(0, 13):
        if middle[l] < target <= middle[l + 1]:
            if l in (0, 12):
                month1, month2 = 12, 1
            else:
                month1, month2 = l, l + 1
            return (month1, month2,
                    middle[l + 1] - target, target - middle[l],
                    middle[l + 1] - middle[l])
    raise WifClimatologyError(f"date {date_str!r} bracketed by no month pair")


def monthly_interp_to_date(monthly, date_str):
    """Temporal weighting in float32, matching real.exe's REAL arithmetic."""
    month1, month2, w1, w2, den = monthly_interp_weights(date_str)
    monthly = np.asarray(monthly, dtype=np.float32)
    f1 = monthly[month1 - 1]
    f2 = monthly[month2 - 1]
    return ((f2 * np.float32(w2) + f1 * np.float32(w1))
            / np.float32(den)).astype(np.float32)


def orient_bottom_up(pressure_now, *fields_now):
    """real.exe's upside-down test and flip (:2367-2372, :2393-2404).

    The decision is made ONCE from the pressure stack -- pressure
    increasing with level index means the data arrived top-down -- and the
    same flip is applied to the pressure and every species, exactly as the
    ``wif_upside_down`` flag drives all three loops.
    """
    nlev = pressure_now.shape[0]
    mid = nlev // 2
    # module_initialize_real.F:2368: p(num/2-1) - p(num/2+1) < 0 at the
    # patch corner; the field is horizontally smooth so the corner value
    # generalizes -- evaluated here at the same single point [0, 0].
    # WRF's indices are only meaningful at its required num_wif_levels=30
    # (the Fortran pair maps to python mid-2 and mid); for a shorter test
    # column they are clamped into range, which preserves the decision --
    # any two distinct levels order the same way on monotonic data.
    low = max(mid - 2, 0)
    high = min(max(mid, low + 1), nlev - 1)
    upside_down = (pressure_now[low, 0, 0]
                   - pressure_now[high, 0, 0]) < 0.0
    if upside_down:
        flipped = tuple(np.ascontiguousarray(f[::-1])
                        for f in (pressure_now,) + fields_now)
        return flipped[0], flipped[1:]
    return pressure_now, fields_now


def vert_interp_wif_column_grid(field_now, pressure_now, target_pressure,
                                *, zap_close_levels=ZAP_CLOSE_LEVELS_PA,
                                force_sfc_in_vinterp=FORCE_SFC_IN_VINTERP):
    """``vert_interp`` for the WIF climatology call, per WRF's exact column.

    ``field_now``/``pressure_now`` are ``(nlev, ny, nx)`` bottom-up
    (pressure decreasing with index); ``target_pressure`` is the DRY
    half-level eta pressure ``c3h(k)*mu0 + c4h(k) + p_top`` ``(nz, ny,
    nx)`` -- what the ``grid%pb`` scratch holds when real.exe makes this
    call (:1625-1629, :1701), identical to the mass target of the met-field
    interpolations.  Ports the
    surface-is-lowest branch of module_initialize_real.F:6035-6086 (level 1
    occupies the surface slot; sigma data has no below-ground levels, so
    ``ko_above_sfc==2`` always takes that branch), the force_sfc and
    zap-close skips, then piecewise linear in LOG(p) with constant
    extrapolation below the deepest assembled point ('Q'/extrap_type=2) and
    a hard refusal above the source top (WRF fatals there too).
    """
    field_now = np.asarray(field_now, dtype=np.float32)
    pressure_now = np.asarray(pressure_now, dtype=np.float32)
    target = np.asarray(target_pressure, dtype=np.float32)
    nlev, ny, nx = pressure_now.shape
    nz = target.shape[0]
    out = np.empty((nz, ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            porig = pressure_now[:, j, i]
            forig = field_now[:, j, i]
            pnew = target[:, j, i]
            ordered_p = [porig[0]]
            ordered_f = [forig[0]]
            # force_sfc_in_vinterp: skip data levels still below the
            # pressure of eta level ``force_sfc_in_vinterp`` (:6053-6066).
            knext = 1
            for ko in range(1, nlev):
                if porig[ko] <= pnew[force_sfc_in_vinterp - 1]:
                    knext = ko
                    break
            else:
                knext = 1
            # Fill above the surface with the cumulative zap-close skip;
            # the topmost level is always kept (:6073-6086).
            for ko in range(knext, nlev):
                if (ordered_p[-1] - porig[ko] < zap_close_levels
                        and ko < nlev - 1):
                    continue
                ordered_p.append(porig[ko])
                ordered_f.append(forig[ko])
            op = np.asarray(ordered_p, dtype=np.float32)
            of = np.asarray(ordered_f, dtype=np.float32)
            if np.any(pnew < op[-1]):
                raise WifClimatologyError(
                    "model eta level lies above the WIF climatology top; "
                    "WRF's lagrange_setup treats the same state as fatal")
            lp = np.log(op)
            lt = np.log(pnew)
            # Piecewise linear in log p; below the deepest point constant.
            idx = np.searchsorted(-lp, -lt, side="left")
            below = idx == 0
            idx = np.clip(idx, 1, lp.size - 1)
            k0 = idx - 1
            weight = (lt - lp[k0]) / (lp[idx] - lp[k0])
            column = of[k0] + weight.astype(np.float32) * (of[idx] - of[k0])
            column[below] = of[0]
            out[:, j, i] = column
    return out



def four_pt_bilinear_rust(field, startlat, deltalat, startlon, deltalon,
                          target_lat, target_lon):
    """:func:`four_pt_bilinear` on the Rust bridge.

    The bridge rebuilds the source axes from ``start``/``delta`` exactly
    as the reference does (``axis[i] = start + delta * i`` in FP64, then
    ``d = axis[1] - axis[0]``) and decides longitude cyclicity from the
    resulting span, so the operator is the same operator -- FP64
    coordinates, FP64 weights, one round to FP32 -- written in Rust.
    """

    backend = _cpu_backend()
    return backend.interpolate_regular_cyclic(
        field, target_lat, target_lon,
        startlat=startlat, deltalat=deltalat,
        startlon=startlon, deltalon=deltalon)


def vert_interp_wif_column_grid_rust(
        field_now, pressure_now, target_pressure, *,
        zap_close_levels=ZAP_CLOSE_LEVELS_PA,
        force_sfc_in_vinterp=FORCE_SFC_IN_VINTERP):
    """:func:`vert_interp_wif_column_grid` on the EXISTING bridge entry.

    No new Rust was written for the vertical stage, and that is the
    point: ``gpuwm_wrf_vert_interp_f32`` already IS WRF's ``vert_interp``
    for every met field, so the WIF call is not a new operator -- it is
    the same operator handed the column WRF hands it.  What the WIF call
    does differently is entirely in the ARGUMENTS, and each one is
    WRF's own:

    * level 1 occupies the SURFACE slot (``module_initialize_real.F``
      :6035-6086), so the bridge takes ``pressure_now[0]``/``field_now[0]``
      as ``surface_pressure``/``surface_value`` and levels 1.. as the
      source stack.  With no data below the surface the bridge takes its
      ``ko_above_sfc == 2`` branch -- the cumulative zap-close loop --
      which is the branch the reference ports.
    * ``vboundb = nz`` forces the LINEAR window at every eta level.  The
      WIF call passes ``linear_interp`` (Registry default 1) in the
      lagrange-order slot (:2452-2465), so the parabolic window the
      met-field calls use above ``vbound`` never applies here.
    * ``extrap="constant"`` is ``extrap_type`` 2 for var_type 'Q';
      a target above the source top stays a hard refusal.
    """

    backend = _cpu_backend()
    field_now = np.asarray(field_now, dtype=np.float32)
    pressure_now = np.asarray(pressure_now, dtype=np.float32)
    target = np.asarray(target_pressure, dtype=np.float32)
    if field_now.shape[0] < 2:
        raise WifClimatologyError(
            "the WIF column needs at least two levels: level 1 occupies "
            "the surface slot, so a single-level source leaves nothing "
            "above it to interpolate between")
    try:
        return backend.wrf_vertical_interpolate(
            np.ascontiguousarray(field_now[1:]),
            np.ascontiguousarray(field_now[0]),
            np.ascontiguousarray(pressure_now[1:]),
            np.ascontiguousarray(pressure_now[0]),
            target,
            interp_in_logp=True, extrap="constant",
            force_sfc_in_vinterp=int(force_sfc_in_vinterp),
            zap_close_levels=float(zap_close_levels),
            vboundb=int(target.shape[0]))
    except ValueError as exc:
        if "target pressure lies above the source top" in str(exc):
            raise WifClimatologyError(
                "model eta level lies above the WIF climatology top; "
                "WRF's lagrange_setup treats the same state as fatal"
            ) from exc
        raise

def wif_surface_emission(w_wif_now_level1, phb):
    """``qnwfa2d`` from the climatology (:4530-4536); ``qnifa2d`` is zero.

    ``z1`` is the thickness of the first model layer from the base-state
    geopotential ``phb`` (staggered, ``(nz+1, ny, nx)``).
    """
    z1 = (np.asarray(phb[1], dtype=np.float32)
          - np.asarray(phb[0], dtype=np.float32)) / np.float32(_G)
    qnwfa2d = (np.asarray(w_wif_now_level1, dtype=np.float32)
               * np.float32(0.000196) * (np.float32(50.0) / z1))
    return qnwfa2d.astype(np.float32), np.zeros_like(qnwfa2d)


def wif_fields_for_grid(climatology: WifClimatology, target_lat, target_lon,
                        date_str, pb, phb, *, backend="rust"):
    """The full pipeline: horizontal -> temporal -> orient -> vertical -> 2d.

    Stage order is WRF's own: metgrid interpolates every monthly level
    horizontally first; real.exe then weights months to ``date_str``,
    orients bottom-up, and vertically interpolates onto the dry eta
    pressure (the ``grid%pb`` scratch of :1701).  Returns
    ``{"nwfa", "nifa", "nwfa2d", "nifa2d"}`` plus a receipt naming the
    stages and their authorities.
    """
    engine = _resolve_backend(backend)
    # The 12 monthly levels interpolated onto this grid do not depend on the
    # date: one climatology object on one grid is interpolated once
    # (:data:`_HORIZONTAL`), keyed by the object and the grid's bytes.
    target_lat_host = np.ascontiguousarray(target_lat)
    target_lon_host = np.ascontiguousarray(target_lon)
    horizontal_key = (
        id(climatology), engine,
        target_lat_host.dtype.str, target_lat_host.shape,
        hashlib.sha256(target_lat_host.tobytes()).hexdigest(),
        target_lon_host.dtype.str, target_lon_host.shape,
        hashlib.sha256(target_lon_host.tobytes()).hexdigest())
    held = _HORIZONTAL.get(horizontal_key)
    if held is not None and held[0] is climatology:
        horiz = _remember(_HORIZONTAL, horizontal_key, held)[1]
    elif engine == "rust":
        startlat, deltalat, startlon, deltalon = \
            climatology.axis_parameters()
        horiz = {
            name: _frozen(four_pt_bilinear_rust(
                getattr(climatology, name), startlat, deltalat,
                startlon, deltalon, target_lat, target_lon))
            for name in ("qnwfa", "qnifa", "pressure")
        }
        _remember(_HORIZONTAL, horizontal_key, (climatology, horiz))
    else:
        horiz = {
            name: _frozen(four_pt_bilinear(getattr(climatology, name),
                                           climatology.latitude,
                                           climatology.longitude,
                                           target_lat, target_lon))
            for name in ("qnwfa", "qnifa", "pressure")
        }
        _remember(_HORIZONTAL, horizontal_key, (climatology, horiz))
    vertical = (vert_interp_wif_column_grid_rust if engine == "rust"
                else vert_interp_wif_column_grid)
    now = {name: monthly_interp_to_date(horiz[name], date_str)
           for name in horiz}
    p_now, (w_now, i_now) = orient_bottom_up(
        now["pressure"], now["qnwfa"], now["qnifa"])
    nwfa = vertical(w_now, p_now, pb)
    nifa = vertical(i_now, p_now, pb)
    nwfa2d, nifa2d = wif_surface_emission(w_now[0], phb)
    month1, month2, w1, w2, den = monthly_interp_weights(date_str)
    receipt = {
        "schema": "wrf-v4.7.1-wif-climatology-ingest-v2",
        "engine": engine,
        "engine_detail": (
            "decode gpuwm_wps_intermediate_read, horizontal "
            "gpuwm_regular_cyclic_bilinear_f32, vertical "
            "gpuwm_wrf_vert_interp_f32 (tools/grib1_bridge)"
            if engine == "rust" else
            "NumPy CPU reference -- the equivalence oracle, not a "
            "shipped data path"),
        "source_path": climatology.source_path,
        "source_grid": [int(climatology.latitude.size),
                        int(climatology.longitude.size)],
        "num_wif_levels": int(climatology.pressure.shape[1]),
        "horizontal": "metgrid four_pt bilinear (METGRID.TBL:885-1150)",
        "temporal": {
            "operator": "monthly_interp_to_date "
                        "(module_initialize_real.F:8029-8095)",
            "month1": month1, "month2": month2,
            "weight1_days": w1, "weight2_days": w2, "denominator_days": den,
        },
        "vertical": "vert_interp 'Q' linear log-p onto dry eta pressure "
                    "(module_initialize_real.F:2452/:2519, :6035-6086)",
        "surface_emission": "qnwfa2d=w_wif_now(:,1,:)*0.000196*(50/z1) "
                            "(module_initialize_real.F:4530-4536)",
    }
    return {"nwfa": nwfa, "nifa": nifa,
            "nwfa2d": nwfa2d, "nifa2d": nifa2d}, receipt


def wif_lowest_number_for_grid(climatology, target_lat, target_lon, date_str):
    """Monthly near-surface water-friendly number for surface emissions.

    The three-dimensional initial and boundary aerosol can come from an
    analysis while WRF's surface source still reads monthly climatology.
    Only that lowest level is mapped, with the existing WIF operators.
    """
    bottom = int(np.argmax(climatology.pressure[0, :, 0, 0]))
    startlat, deltalat, startlon, deltalon = climatology.axis_parameters()
    monthly = four_pt_bilinear_rust(
        climatology.qnwfa[:, bottom:bottom + 1],
        startlat, deltalat, startlon, deltalon, target_lat, target_lon)
    return monthly_interp_to_date(monthly, date_str)[0]


# ---------------------------------------------------------------------------
# Asset resolution.  This is the half that makes the dataset the DEFAULT.
#
# The precedent is woof/core/thompson_aerosol_contract.py's
# resolve_ccn_activation_path: an ordered candidate list, an explicit
# operator override that is an ERROR when it does not exist rather than a
# silent demotion, and one named outcome.  Two things differ here, and both
# differences are deliberate:
#
#   * There is no packaged copy to fall back to.  CCN_ACTIVATE.BIN is 12 kB
#     and woof redistributes it; QNWFA_QNIFA_SIGMA_MONTHLY.dat is 225 MB and
#     this repository does not ship it.  So "not found" is a REACHABLE
#     outcome on a correctly installed tree, not a broken install.
#   * "Not found" therefore returns a resolution instead of raising.  WRF
#     itself has no fallback -- real.exe FATALs -- but WRF's own
#     microphysics does: thompson_init installs a synthetic CCN/IN profile
#     when the fields arrive empty (module_mp_thompson.F:493-551).  That
#     profile is a real, WRF-authored initial condition, so falling back to
#     it is defensible; falling back SILENTLY is not, which is why the
#     resolution carries the reason and every candidate it tried.
#
# The cwd candidate is not a convenience.  It is WRF's own locating rule:
# metgrid reads this dataset through constants_name by bare relative name
# from the run directory, exactly as table_ccnAct OPENs CCN_ACTIVATE.BIN.
# A user who assembled a WRF run directory and pointed ArWen at it gets the
# same file WRF would have read, with no new configuration to discover.

#: The dataset's distributed filename.  WRF names it in metgrid's
#: ``constants_name`` and ships it from the WRF download page as
#: ``QNWFA_QNIFA_SIGMA_MONTHLY.dat``; the copy this port was measured
#: against is WRF 4.7.1's.
WIF_CLIMATOLOGY_FILE = "QNWFA_QNIFA_SIGMA_MONTHLY.dat"

#: Point ArWen at one dataset FILE (highest precedence after an explicit
#: RunConfig path).
WIF_CLIMATOLOGY_PATH_ENV = "WOOF_WIF_CLIMATOLOGY"

#: Point ArWen at a DIRECTORY holding it -- typically a WRF ``run/``.
WIF_CLIMATOLOGY_ROOT_ENV = "WOOF_WIF_CLIMATOLOGY_ROOT"

#: Measured properties of the WRF 4.7.1 dataset.  NOT enforced: a user may
#: legitimately hold a different WRF release's copy, and refusing it would
#: substitute a pin for the thing the pin is a proxy for.  Reported in the
#: receipt instead, so a run that used a different file says so.
WIF_REFERENCE_BYTES = 225443520
WIF_REFERENCE_SHA256 = (
    "2f828eabd96a45f3872390f901240ea2259a1e9a629247010f42ce7a31cc46be")
WIF_REFERENCE_SOURCE = "WRF 4.7.1 run/QNWFA_QNIFA_SIGMA_MONTHLY.dat"


def is_wif_intermediate_file(path) -> bool:
    """True when ``path`` BEGINS with a complete IFV=5 field record.

    Presence is not the question the precondition means to ask.  The gate
    that decides whether an externally forced mp=28 domain may launch used
    to ask ``is_file()`` alone, so a zero-byte file -- which is what a failed
    download, an interrupted copy or a placeholder leaves behind --
    satisfied it, and the run then died in the ingest lane with "WPS
    intermediate file holds no records".  A refusal that fires after step 0
    is the shape the gate law forbids, and it fired here because the check
    at step 0 measured the wrong thing.

    What is measured is the FIRST record group, read with this module's own
    reader: the IFV version record, the header it sizes fields out of, the
    projection record, the wind-relative flag and a data record of exactly
    ``4 * nx * ny`` bytes.  That is a few hundred kilobytes on WRF's own
    copy and it is the same decode the ingest performs, so a file that
    passes here cannot fail the ingest's first record.  It is deliberately
    NOT a byte or digest comparison against
    :data:`WIF_REFERENCE_BYTES`/:data:`WIF_REFERENCE_SHA256`: a user may
    hold a different WRF release's copy, and the receipt content-addresses
    whatever was read.  Structure is the property every legitimate copy
    shares.
    """

    try:
        with open(path, "rb") as handle:
            version_record = _read_record(handle)
            if version_record is None or len(version_record) != 4:
                return False
            (version,) = struct.unpack(">i", version_record)
            if version != 5:
                return False
            header = _read_record(handle)
            if header is None or len(header) < 156:
                return False
            _xlvl, nx, ny, _iproj = struct.unpack(">fiii", header[140:156])
            if nx <= 0 or ny <= 0:
                return False
            projection = _read_record(handle)
            if projection is None or len(projection) < 28:
                return False
            if _read_record(handle) is None:
                return False
            data = _read_record(handle)
            return data is not None and len(data) == 4 * nx * ny
    except (OSError, struct.error, WifClimatologyError, ValueError):
        return False


# ONE exception class, not two.  development branch landed a second
# MissingWifClimatologyDataset in woof/ingest/wif_dataset.py.  That would
# have made `except MissingWifClimatologyDataset` catch the refusal from
# one module and let the identical refusal from the other escape -- and a
# single mp=28 run reaches both.  The ASSET module owns it: it is the
# lighter of the two and the one a preflight imports.  This module
# re-exports it, so every existing `from woof.ingest.wif_climatology
# import MissingWifClimatologyDataset` resolves to the same class object.
#
# It means here what it always meant: an EXPLICITLY named dataset is not
# there.  An unset default that finds nothing is not this error; it is a
# fallback, and it returns a :class:`WifSourceResolution` that says so.
from woof.ingest.wif_dataset import (      # noqa: E402
    WIF_DATASET_ROOT_ENV,
    MissingWifClimatologyDataset,
    resolve_wif_data_root,
)


@dataclass(frozen=True)
class WifSourceResolution:
    """Where the mp=28 aerosol initial state is going to come from.

    ``path`` is the dataset when one resolved and ``None`` when none did.
    ``origin`` names the rule that decided, and ``fallback_reason`` is the
    sentence a receipt prints when ``path`` is ``None`` -- populated on
    exactly the ``None`` case, so a consumer cannot forget to say why.

    ``portable_reason`` is the same sentence with each searched location
    named by its rule rather than by this process's absolute path (the
    working directory, the home directory).  It is what a preparation
    receipt binds; see :func:`portable_wif_statement`.
    """

    path: object              # pathlib.Path | None
    origin: str
    candidates: tuple
    fallback_reason: object = None   # str | None
    portable_reason: object = None   # str | None

    @property
    def resolved(self) -> bool:
        return self.path is not None


def resolve_wif_climatology(path=None, *, env=None, cwd=None,
                            explicit_required=False, requested_by=None):
    """Locate ``QNWFA_QNIFA_SIGMA_MONTHLY.dat``, or name why there is none.

    Precedence, highest first:

    1. ``path`` -- ``RunConfig.wif_climatology_path`` (the TOML's
       ``[shared] wif_climatology_path``, which is how the refusals below
       spell it), when non-empty.
    2. ``$WOOF_WIF_CLIMATOLOGY`` -- one file.
    3. ``$WOOF_WIF_CLIMATOLOGY_ROOT`` / ``QNWFA_QNIFA_SIGMA_MONTHLY.dat``.
    4. ``<cwd>/QNWFA_QNIFA_SIGMA_MONTHLY.dat`` -- WRF's own rule.

    Cases 1-3 were chosen by a human.  If a chosen path is absent this
    raises, for the reason the CCN resolver states: silently ignoring an
    operator's override is how a run ends up using an initial condition
    nobody chose.  Case 4 is a probe, so its absence is a fallback.

    ``explicit_required=True`` additionally turns "found nothing at all"
    into the same hard error -- that is what an explicit
    ``mp28_aerosol_source='climatology'`` means, and it is the difference
    between a default that may degrade and a request that is honoured.

    ``requested_by`` names WHICH selector made that request, because there
    are two spellings of it and only one of them is this field: a namelist
    carrying ``(aer_init_opt, wif_input_opt) = (1, 1)`` resolves to the
    same strict request with ``mp28_aerosol_source`` left at its default,
    and a refusal that named the field there would send the operator to
    change a setting they never set.  Defaults to the field, which is what
    every caller that does not say otherwise is asking on behalf of.
    """
    import os
    from pathlib import Path

    environ = os.environ if env is None else env
    base = Path.cwd() if cwd is None else Path(cwd)
    tried = []

    chosen = None
    origin = ""
    if path:
        chosen, origin = Path(path), "[shared] wif_climatology_path"
    elif environ.get(WIF_CLIMATOLOGY_PATH_ENV):
        chosen = Path(environ[WIF_CLIMATOLOGY_PATH_ENV])
        origin = "$" + WIF_CLIMATOLOGY_PATH_ENV
    elif environ.get(WIF_CLIMATOLOGY_ROOT_ENV):
        chosen = Path(environ[WIF_CLIMATOLOGY_ROOT_ENV]) / WIF_CLIMATOLOGY_FILE
        origin = "$" + WIF_CLIMATOLOGY_ROOT_ENV + "/" + WIF_CLIMATOLOGY_FILE
    if chosen is not None:
        tried.append(str(chosen))
        if is_wif_intermediate_file(chosen):
            return WifSourceResolution(chosen, origin, tuple(tried))
        if chosen.is_file():
            raise MissingWifClimatologyDataset(
                origin + " names " + str(chosen) + ", which does not begin "
                "with a WPS intermediate (IFV=5) field record, so it is not "
                "the WIF aerosol climatology -- a truncated or empty "
                "download is the usual cause. Re-fetch it with `woof "
                "fetch-tables --wif`, or point the setting at WRF's "
                + WIF_CLIMATOLOGY_FILE + " ("
                + str(WIF_REFERENCE_BYTES) + " bytes, " + WIF_REFERENCE_SOURCE
                + ").")
        raise MissingWifClimatologyDataset(
            origin + " names " + str(chosen) + ", and there is no such file. "
            "Re-fetch it with `woof fetch-tables --wif`, or point the "
            "setting at WRF's " + WIF_CLIMATOLOGY_FILE + " ("
            + str(WIF_REFERENCE_BYTES) + " bytes, " + WIF_REFERENCE_SOURCE
            + ").")

    # THE STAGED ROOT (development branch).  `woof fetch-tables
    # --wif` installs the dataset into $WOOF_WIF_DATA_ROOT, defaulting to
    # ~/.woof/wif, verified against the pin before it is moved into place.
    # Without this rung that command would stage a 225 MB file the DEFAULT
    # path never looks at: the asset route would be present and
    # disconnected from the thing it exists to feed.  Searched BEFORE the
    # working directory, because it was put there on purpose by the command
    # this refusal recommends while the cwd candidate is a probe; and AFTER
    # the two human-chosen overrides, because those were typed for this run.
    staged = resolve_wif_data_root(None, env=environ) / WIF_CLIMATOLOGY_FILE
    tried.append(str(staged))
    if is_wif_intermediate_file(staged):
        return WifSourceResolution(
            staged, "staged by `woof fetch-tables --wif`", tuple(tried))

    probe = base / WIF_CLIMATOLOGY_FILE
    tried.append(str(probe))
    if is_wif_intermediate_file(probe):
        return WifSourceResolution(
            probe,
            "working directory (WRF constants_name rule: bare relative "
            + WIF_CLIMATOLOGY_FILE + ")",
            tuple(tried))

    reason = _not_found_sentence(str(staged), str(probe))
    if explicit_required:
        raise MissingWifClimatologyDataset(
            (requested_by or "mp28_aerosol_source='climatology'")
            + " requires the dataset and " + reason)
    # The portable spelling names the two default locations by rule.  The
    # absolute spelling above is this process's cwd and home, and the
    # namelist import used to write it into the WRF-input door's
    # preparation receipt: a --restart launched from another directory
    # then hashed a different receipt and was refused as "written for a
    # different run" (5 km radar DA, 2026-10-23), though nothing about the
    # run differed.
    portable = _not_found_sentence(
        "$" + WIF_DATASET_ROOT_ENV + ", default ~/.woof/wif",
        "./" + WIF_CLIMATOLOGY_FILE)
    return WifSourceResolution(None, "unresolved", tuple(tried), reason,
                               portable)


def _download_wif_dataset(root) -> int:
    """`woof fetch-tables --wif` into ``root``: 0 staged, 2 refused."""
    from woof.table_assets import stage_wif_dataset

    return stage_wif_dataset(None, root)


def acquire_wif_climatology(path=None, *, env=None, cwd=None,
                            requested_by=None):
    """Locate the dataset, download it when it is not here, or refuse by name.

    What a run that COMMITS to the climatology calls (``mp28_aerosol_source``
    'auto' or 'climatology' on a real-data mp=28 preparation).  Named
    breakage (a real-data mp=28 run, 2026-10-23
    12:58Z): with no copy on the machine, 'auto' used to fall back to
    thompson_init's synthetic CCN/IN profile with only a warning, so a run
    on a box missing one 225 MB file was a different forecast from the same
    run on a box that had it, and nothing failed.  Now the search below is
    followed by the pinned download `woof fetch-tables --wif` performs
    (exact size and SHA-256 before the atomic install into
    ``$WOOF_WIF_DATA_ROOT``, default ~/.woof/wif), and when that fails too
    the run is refused with the ways out.  The synthetic profile is reached
    only by naming it (``mp28_aerosol_source='synthetic'``).

    A path a human chose that is not the dataset raises exactly as
    :func:`resolve_wif_climatology` does; it is never replaced by a download.
    """

    resolution = resolve_wif_climatology(path, env=env, cwd=cwd,
                                         requested_by=requested_by)
    if resolution.resolved:
        return resolution
    import os
    import sys
    from pathlib import Path

    environ = os.environ if env is None else env
    root = resolve_wif_data_root(None, env=environ)
    print(f"mp_physics=28 aerosol: {WIF_CLIMATOLOGY_FILE} is not on this "
          f"machine; downloading the pinned copy into {root}", file=sys.stderr)
    if _download_wif_dataset(root) == 0:
        resolution = resolve_wif_climatology(path, env=env, cwd=cwd,
                                             requested_by=requested_by)
        if resolution.resolved:
            return resolution
    searched = str(resolution.fallback_reason or _not_found_sentence(
        str(root / WIF_CLIMATOLOGY_FILE),
        str(Path.cwd() / WIF_CLIMATOLOGY_FILE))).rstrip()
    if not searched.endswith("."):
        searched += "."
    raise MissingWifClimatologyDataset(
        (requested_by or "mp28_aerosol_source='auto'")
        + " needs WRF's monthly WIF aerosol climatology to initialize "
        "mp_physics=28, and " + searched
        + " The pinned download into " + str(root) + " failed as well "
        "(its reason is printed above). Without the dataset the aerosol "
        "would start from thompson_init's synthetic CCN/IN profile, a "
        "different forecast from WRF's, so the run stops here. Stage it "
        "with `woof fetch-tables --wif` (or `--wif --from DIR` offline), "
        "set [shared] wif_climatology_path or $" + WIF_CLIMATOLOGY_PATH_ENV
        + ", or select [shared] mp28_aerosol_source='synthetic' to take the "
        "synthetic profile deliberately.")


def _not_found_sentence(staged: str, probe: str) -> str:
    return (
        "no " + WIF_CLIMATOLOGY_FILE + " was found. Searched, in order: "
        "[shared] wif_climatology_path (unset), $" + WIF_CLIMATOLOGY_PATH_ENV
        + " (unset), $" + WIF_CLIMATOLOGY_ROOT_ENV + " (unset), the root "
        "`woof fetch-tables --wif` stages into (" + staged + "), and "
        "the working directory (" + probe + "). A candidate that "
        "exists but does not begin with a complete WPS intermediate "
        "(IFV=5) field record is not counted as found, because an empty "
        "or truncated copy would refuse in the ingest lane instead of "
        "here. WOOF does not "
        "redistribute this 225 MB dataset; it is WRF's own, downloaded with "
        "WRF, taken from a WRF run/ directory (" + WIF_REFERENCE_SOURCE
        + "), or staged with `woof fetch-tables --wif --from DIR`.")


def _file_sha256(path) -> str:
    """SHA-256 of a resolved dataset, cached per (path, size, mtime)."""
    from pathlib import Path

    stat = Path(path).stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _SHA256_CACHE.get(key)
    if cached is None:
        import hashlib

        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while True:
                block = stream.read(1 << 22)
                if not block:
                    break
                digest.update(block)
        cached = _SHA256_CACHE[key] = digest.hexdigest()
    return cached


_SHA256_CACHE: dict = {}


def portable_wif_statement(resolution) -> str:
    """The resolution as a preparation receipt binds it: by content, not by host path.

    A resolved dataset is named by the rule that found it, its byte count
    and its SHA-256; an unresolved search by :attr:`WifSourceResolution.
    portable_reason`.  Two processes that would read the same bytes (or
    find none) from different working directories or home directories
    state the same sentence, and two that would read different copies do
    not.  The absolute paths stay in what a run prints and in
    :func:`describe_wif_source`, which are records, not identity.
    """
    if resolution.resolved:
        from pathlib import Path

        sha256 = _file_sha256(resolution.path)
        size = Path(resolution.path).stat().st_size
        return (f"{resolution.origin}; {size} bytes, sha256 {sha256}"
                + ("" if sha256 != WIF_REFERENCE_SHA256
                   else f" (the {WIF_REFERENCE_SOURCE} reference)"))
    return str(resolution.portable_reason or resolution.fallback_reason or "")


def describe_wif_source(resolution):
    """Content-address whatever was resolved, for the run receipt.

    The digest is computed, not assumed: two users with two WRF releases'
    copies produce two different initial conditions, and a receipt that
    merely repeated the pinned constant would hide exactly that.
    """
    from pathlib import Path

    if not resolution.resolved:
        return {"resolved": False, "origin": resolution.origin,
                "candidates": list(resolution.candidates),
                "fallback_reason": resolution.fallback_reason}
    source = Path(resolution.path)
    sha256 = _file_sha256(source)
    return {
        "resolved": True,
        "origin": resolution.origin,
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256,
        "matches_wrf_471_reference": sha256 == WIF_REFERENCE_SHA256,
        "reference": WIF_REFERENCE_SOURCE,
    }
