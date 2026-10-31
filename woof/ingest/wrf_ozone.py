"""Exact-bit port of WRF v4.6.1's o3input=2 ozone pipeline (upstream of RRTMG).

This module reproduces, bit for bit under WRF's FP32 build (RWORDSIZE=4),
the chain that produces the radiation driver's ``o3rad`` / ``O33D`` field
for the legacy RRTMG option-4 schemes:

1. ``load_ozone_climatology`` -- the read of ``ozone_plev.formatted``,
   ``ozone_lat.formatted`` and ``ozone.formatted`` performed by
   ``oznini`` (phys/module_ra_cam_support.F), including the ``plev*100.``
   hPa->Pa conversion.  One list-directed value per record; months are
   read into slots 2..13 of WRF's ``ozmixin`` (slot 1 is never written).
2. ``interp_ozone_to_latitudes`` -- oznini's per-point latitude
   interpolation ``lin_interpol2`` (module_ra_cam_support.F), which
   linearly EXTRAPOLATES beyond the 64 climatology latitudes (+-87.8638
   deg) using the edge-interval slope; there is no clamping.
3. ``ozn_time_int`` (phys/module_radiation_driver.F) -- time
   interpolation to the current Julian day with December-January wrap
   (``ozncyc = .true.``; the ``julday`` argument is declared but unused
   by WRF, and is accepted-and-ignored here for signature fidelity).
4. ``ozn_p_int`` (phys/module_radiation_driver.F) -- pressure
   interpolation from the 59 climatology levels onto model layer
   midpoints, with the CAM ``kupper`` bracket-carrying algorithm,
   top-of-data scaling ``ozmixt(1)*p/pin(1)`` and constant extension
   below ``pin(levsiz)``.

``o33d_profile`` chains 1-4 exactly as the driver does for a batch of
columns.

PRECISION (verified against the compiled control; see tests)
------------------------------------------------------------
Under the FP32 build every stage of this pipeline executes in default
REAL (float32): the oznini read targets, ``lin_interpol2`` (r4 by
design: "J. Done changed from r8 to r4"), ``ozn_time_int`` and
``ozn_p_int`` declare plain ``real`` throughout.  No r8 arithmetic
exists anywhere on this path, so every intermediate here is float32
with one IEEE rounding per Fortran operation.

UNITS AND THE O33D BOUNDARY (frozen contract for the wrapper lanes)
-------------------------------------------------------------------
* ``pin``/``plev`` are Pa (the file stores hPa; oznini multiplies by
  100. in FP32).
* ``p`` supplied by the radiation driver to ``ozn_p_int`` is the layer
  midpoint pressure P3D in Pa, bottom-up (NOT p8w, NOT hPa).
* O33D == ``o3rad`` == the ``o3vmr`` output of ``ozn_p_int``: the
  climatology mixing-ratio values themselves, interpolated.  NO unit
  conversion happens anywhere on the o3input=2 path -- not in the
  driver, not in ``ozn_time_int``/``ozn_p_int``, and not in the
  RRTMG_LWRAD/RRTMG_SWRAD wrappers, which copy ``O31D(K)=O33D(I,K,J)``
  straight into their ``o3vmr(ncol,K)`` for K <= kte and hand it to
  RRTMG as volume mixing ratio (``wkl(3,:)``).  The wrappers' local
  ``amdo = 0.603461`` is applied ONLY to the built-in ``inirad`` mmr
  profile: for o3input /= 2, and above the model top (k > kte) where
  the climatology profile is shifted onto ``o31d(kte)``.  (Registry
  labels o3rad "ppmv"; the actual values are absolute mixing ratio,
  ~1e-6.)
"""

from __future__ import annotations

from functools import lru_cache
import ctypes
from pathlib import Path
from typing import NamedTuple

import numpy as np

from woof.file_digest import sha256_file as _sha256


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "wrf_radiation"

#: SHA-256 of run/ozone*.formatted from the WRF v4.6.1 source authority
#: (see woof/data/wrf_radiation/PROVENANCE.md).
OZONE_SHA256 = \
    "bcd5316d213625cf28805a83dd6703cdebc836422c9c89ae7ec0fccf4d0c4389"
OZONE_LAT_SHA256 = \
    "00ca601bf61c40ae06fccb692ff3d1ab98a60be581892ec2c8a168d58512aadb"
OZONE_PLEV_SHA256 = \
    "5c9cc9855b12f699f25213a351f8cb7526dcb31b784cdddd9a951d2578b56b3a"

LEVSIZ = 59   #: climatology pressure levels (module_check_a_mundo for RRTMG)
LATSIZ = 64   #: climatology latitudes (oznini local parameter)

#: ozn_time_int's date_oz: mid-month Julian day of each monthly sample.
DATE_OZ = (16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350)

_F32 = np.float32


class OzoneClimatology(NamedTuple):
    """Parsed climatology exactly as oznini holds it (all float32).

    plev:  (59,) Pa, top-down (strictly increasing); equals WRF's
           ``plev``/``pin`` after the *100. conversion.
    lat:   (64,) degrees, south-to-north (strictly increasing).
    ozmix: (59, 64, 12) mixing ratio; ``ozmix[k-1, j-1, m-2]`` equals
           WRF's ``ozmixin(1, k, j, m)`` (m = 2..13 -> Jan..Dec).
    """

    plev: np.ndarray
    lat: np.ndarray
    ozmix: np.ndarray


def _read_formatted(path: Path, count: int, label: str) -> np.ndarray:
    """One list-directed READ per record, into default REAL (float32).

    WRF executes ``READ (unit,*) x(k)`` inside a DO loop, so each value
    comes from its own record; the packaged files hold exactly one value
    per line.  gfortran's runtime converts the decimal text with correct
    rounding to float32; each token here is converted the same way (the
    oracle gate proves the parses agree bit for bit).
    """
    from woof.tables import table_path
    path = table_path("ozone", path)
    values = []
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            token = line.strip()
            if not token:
                continue
            values.append(_F32(token))
    if len(values) != count:
        raise ValueError(
            f"{label}: expected {count} records, found {len(values)}")
    return np.array(values, dtype=np.float32)


@lru_cache(maxsize=4)
def _load_ozone_climatology_cached(data_dir: str | None) -> OzoneClimatology:
    base = DATA_DIR if data_dir is None else Path(data_dir)
    paths = {
        "ozone_plev.formatted": (OZONE_PLEV_SHA256, LEVSIZ),
        "ozone_lat.formatted": (OZONE_LAT_SHA256, LATSIZ),
        "ozone.formatted": (OZONE_SHA256, LEVSIZ * LATSIZ * 12),
    }
    if data_dir is None:
        for name, (sha, _) in paths.items():
            from woof.tables import table_path
            digest = _sha256(table_path("ozone", base / name))
            if digest != sha:
                raise ValueError(
                    f"packaged {name} checksum mismatch: {digest} != {sha}")

    plev = _read_formatted(base / "ozone_plev.formatted", LEVSIZ,
                           "ozone_plev.formatted")
    # oznini: plev(k) = plev(k)*100.  (hPa -> Pa, FP32)
    plev = (plev * _F32(100.0)).astype(np.float32)
    lat = _read_formatted(base / "ozone_lat.formatted", LATSIZ,
                          "ozone_lat.formatted")
    flat = _read_formatted(base / "ozone.formatted", LEVSIZ * LATSIZ * 12,
                           "ozone.formatted")
    # Read order: m (outer, 12 months) -> j (64 lats) -> k (59 levels).
    ozmix = flat.reshape(12, LATSIZ, LEVSIZ).transpose(2, 1, 0).copy()

    if not np.all(np.diff(plev) > 0):
        raise ValueError("ozone_plev.formatted is not strictly increasing")
    if not np.all(np.diff(lat) > 0):
        raise ValueError("ozone_lat.formatted is not strictly increasing")
    if not np.isfinite(ozmix).all() or (ozmix <= 0).any():
        raise ValueError("ozone.formatted has non-finite or non-positive "
                         "mixing ratios")
    for arr in (plev, lat, ozmix):
        arr.setflags(write=False)
    return OzoneClimatology(plev=plev, lat=lat, ozmix=ozmix)


def load_ozone_climatology(data_dir: str | Path | None = None
                           ) -> OzoneClimatology:
    """Load the CAM ozone climatology exactly as WRF's oznini reads it.

    With ``data_dir=None`` the packaged copies under
    ``woof/data/wrf_radiation`` are used and their pinned SHA-256
    digests are enforced.
    """
    return _load_ozone_climatology_cached(
        None if data_dir is None else str(Path(data_dir)))


def _native_ozone_entries():
    """Optional additive ABI; older bridges retain the original host path."""
    # Exception/callback/log/print policies and requested underflow warnings
    # are observable behavior of the original array arithmetic.
    policy = np.geterr()
    if policy["under"] != "ignore" or any(mode not in ("ignore", "warn") for mode in policy.values()):
        return None
    from woof.core import portable_math
    try:
        library = portable_math._load()
    except (OSError, RuntimeError, FileNotFoundError):
        return None
    names = ("gpuwm_ozone_validate_latitudes_f32", "gpuwm_ozone_latitude_f32",
             "gpuwm_ozone_latitude_time_f32")
    if library is None or not all(hasattr(library, name) for name in names):
        return None
    entries = tuple(getattr(library, name) for name in names)
    pointer, size, f32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_float
    entries[0].argtypes = [pointer, size]
    common = [pointer, size, pointer, size, pointer, size, size]
    entries[1].argtypes = common + [size, pointer]
    entries[2].argtypes = common + [size, size, f32, f32, size, pointer]
    for entry in entries:
        entry.restype = ctypes.c_int32
    return entries


def _native_xlat(entries, values):
    values = np.ascontiguousarray(values.reshape(-1))
    status = entries[0](values.ctypes.data, values.size)
    if status == 3:
        raise ValueError("XLAT contains non-finite values")
    if status != 0:
        raise RuntimeError(f"native ozone latitude validation failed ({status})")
    return values


def _native_climatology(climo):
    # Metadata and transport only. Unusual external climatologies keep their
    # original NumPy conversions, shape errors and arithmetic.
    latitude, ozone = climo.lat, climo.ozmix
    if (not isinstance(latitude, np.ndarray) or latitude.dtype != np.float32
            or latitude.ndim != 1 or latitude.size < 2
            or not isinstance(ozone, np.ndarray) or ozone.dtype != np.float32
            or ozone.shape != (LEVSIZ, latitude.size, 12)):
        return None
    return np.ascontiguousarray(latitude), np.ascontiguousarray(ozone)


def _native_ozone_status(status):
    if status == 3:
        raise ValueError("XLAT contains non-finite values")
    if status not in (0, 126):
        raise RuntimeError(f"native ozone interpolation failed ({status})")
    return status == 0


def interp_ozone_to_latitudes(xlat, climo: OzoneClimatology | None = None
                              ) -> np.ndarray:
    """oznini's latitude interpolation onto arbitrary XLAT points.

    Statement-faithful ``lin_interpol2(lat, ozmixin(1,k,:,m), xlat)`` per
    (point, level, month), in FP32:

    * bracket k: smallest interval with ``y <= x(k+1)``, clamped to the
      edge intervals for y outside [x(1), x(n)] -- so out-of-range
      latitudes are linearly EXTRAPOLATED with the edge slope;
    * ``a = (f(k+1)-f(k)) / (x(k+1)-x(k))``; ``g = f(k) + a*(y-x(k))``,
      one FP32 rounding per operation.

    Returns float32 with shape ``xlat.shape + (59, 12)`` (months
    Jan..Dec, i.e. WRF's ozmixm slots 2..13).
    """
    if climo is None:
        climo = load_ozone_climatology()
    y = np.asarray(xlat, dtype=np.float32)
    entries = _native_ozone_entries()
    if entries is not None:
        values = _native_xlat(entries, y)
        inputs = _native_climatology(climo)
        if inputs is not None:
            from woof.core import portable_math
            latitude, ozone = inputs
            out = np.empty(y.shape + (LEVSIZ, 12), dtype=np.float32)
            status = entries[1](values.ctypes.data, values.size,
                                latitude.ctypes.data, latitude.size,
                                ozone.ctypes.data, LEVSIZ, 12,
                                portable_math._workers(None), out.ctypes.data)
            if _native_ozone_status(status):
                return out
    if not np.isfinite(y).all():
        raise ValueError("XLAT contains non-finite values")
    shape = y.shape
    y = y.ravel()
    x = climo.lat
    n = x.size
    # WRF: k=1 if y<=x(1); k=n-1 if y>=x(n); else first k with y<=x(k+1).
    k = np.searchsorted(x, y, side="left").astype(np.int64) - 1
    k = np.clip(k, 0, n - 2)
    f = climo.ozmix                       # (59, 64, 12)
    fk = f[:, k, :]                       # (59, npts, 12)
    fk1 = f[:, k + 1, :]
    a = (fk1 - fk) / (x[k + 1] - x[k])[None, :, None]
    g = fk + a * (y - x[k])[None, :, None]
    out = np.ascontiguousarray(np.moveaxis(g.astype(np.float32), 1, 0))
    return out.reshape(shape + (LEVSIZ, 12))


def ozn_time_int(julday, julian, ozmixm) -> np.ndarray:
    """Transcription of ozn_time_int (phys/module_radiation_driver.F).

    julday : accepted and ignored -- WRF declares it INTENT(IN) and never
             reads it.
    julian : REAL day-of-year, 0.0 at 0Z on 1 Jan (converted to FP32).
    ozmixm : float32 (..., 59, 12), months Jan..Dec on the last axis
             (WRF slots 2..13; slot 1 does not exist here because WRF
             never writes or reads it).

    Returns ``ozmixt`` float32 (..., 59).
    """
    del julday  # unused, exactly as in WRF
    ozmixm = np.asarray(ozmixm)
    if ozmixm.dtype != np.float32 or ozmixm.shape[-1] != 12:
        raise ValueError("ozmixm must be float32 with a trailing "
                         "12-month axis")
    nm, np_, fact1, fact2 = _ozone_time_weights(julian)
    return ozmixm[..., nm] * fact1 + ozmixm[..., np_] * fact2


def _ozone_time_weights(julian):
    """Calendar scalars in the original WRF float32 statement order."""
    intjulian = _F32(_F32(julian) + _F32(1.0))
    ijul = int(intjulian)                       # Fortran INT(): truncation
    intjulian = _F32(intjulian - _F32(ijul))    # FLOAT(IJUL) then subtract
    ijul = ijul % 365
    if ijul == 0:
        ijul = 365
    intjulian = _F32(intjulian + _F32(ijul))
    np1 = 1
    finddate = False
    for m in range(1, 13):
        if _F32(DATE_OZ[m - 1]) > intjulian and not finddate:
            np1 = m
            finddate = True
    cdayozp = _F32(DATE_OZ[np1 - 1])
    if np1 > 1:
        cdayozm = _F32(DATE_OZ[np1 - 2])
        np_ = np1
        nm = np_ - 1
    else:
        cdayozm = _F32(DATE_OZ[11])
        np_ = np1
        nm = 12
    daysperyear = _F32(365.0)
    if np1 == 1:                       # ozncyc: Dec-Jan interpolation
        deltat = _F32(_F32(cdayozp + daysperyear) - cdayozm)
        if intjulian > cdayozp:        # December
            fact1 = _F32(_F32(_F32(cdayozp + daysperyear) - intjulian)
                         / deltat)
            fact2 = _F32(_F32(intjulian - cdayozm) / deltat)
        else:                          # January
            fact1 = _F32(_F32(cdayozp - intjulian) / deltat)
            fact2 = _F32(_F32(_F32(intjulian + daysperyear) - cdayozm)
                         / deltat)
    else:
        deltat = _F32(cdayozp - cdayozm)
        fact1 = _F32(_F32(cdayozp - intjulian) / deltat)
        fact2 = _F32(_F32(intjulian - cdayozm) / deltat)
    # WRF: ozmixt = ozmixm(:,:,:,nm+1)*fact1 + ozmixm(:,:,:,np+1)*fact2;
    # slots nm+1/np+1 (2..13) are months nm/np (1..12) -> 0-based nm-1/np-1.
    return nm - 1, np_ - 1, fact1, fact2


def ozn_latitude_time_int(julday, julian, xlat,
                         climo: OzoneClimatology | None = None, *,
                         out: np.ndarray | None = None) -> np.ndarray:
    """The exact latitude then time chain, computing only its two months.

    Return float32 ``xlat.shape + (59,)``. The native route rounds each
    latitude statement before the two separate products and monthly sum.
    An older bridge or unusual external climatology takes the original chain.

    ``out``, a C-contiguous float32 array of the result's shape, is filled
    and returned by the native route instead of a fresh array: a radiation
    adapter reuses one buffer every call, so a 1 km rank does not fault in
    ~200 MB of new host memory at every radiation step (A3).
    """
    del julday  # WRF accepts and ignores it.
    if climo is None:
        climo = load_ozone_climatology()
    y = np.asarray(xlat, dtype=np.float32)
    entries = _native_ozone_entries()
    if entries is not None:
        values = _native_xlat(entries, y)
        inputs = _native_climatology(climo)
        if inputs is not None:
            from woof.core import portable_math
            latitude, ozone = inputs
            nm, np_, fact1, fact2 = _ozone_time_weights(julian)
            shape = y.shape + (LEVSIZ,)
            if (out is None or out.shape != shape or out.dtype != np.float32
                    or not out.flags.c_contiguous):
                out = np.empty(shape, dtype=np.float32)
            status = entries[2](values.ctypes.data, values.size,
                                latitude.ctypes.data, latitude.size,
                                ozone.ctypes.data, LEVSIZ, 12, nm, np_, fact1, fact2,
                                portable_math._workers(None), out.ctypes.data)
            if _native_ozone_status(status):
                return out
    return ozn_time_int(None, julian, interp_ozone_to_latitudes(y, climo))


def ozn_p_int(p, pin, ozmixt) -> np.ndarray:
    """Transcription of ozn_p_int (phys/module_radiation_driver.F).

    p      : float32 (ncol, nz) layer-midpoint pressures, Pa, BOTTOM-UP
             (index 0 = lowest model layer), exactly as the radiation
             driver passes its ``p`` (P3D) tile.
    pin    : float32 (59,) climatology level pressures, Pa, top-down
             (``OzoneClimatology.plev``); must be strictly increasing
             (WRF aborts on suspected non-monotonic data).
    ozmixt : float32 (ncol, 59) time-interpolated mixing ratios.

    Returns ``o3vmr`` float32 (ncol, nz), bottom-up like ``p``.

    The WRF loop nest carries ``kupper`` from level to level and has an
    all-columns-bracketed early exit; because ``pin`` is strictly
    increasing and each column's pressure increases as the loop walks
    down, a column's bracket index never lies below any previous
    ``kupper`` minimum, and the early-exit interpolation is the same
    expression as the fall-through ``else`` branch.  The port below is
    therefore statement-equivalent while vectorizing over columns; the
    one behavioral subtlety kept faithfully is that a level exactly
    equal to ``pin(1)`` matches no bracket and interpolates with the
    column's STALE ``kupper`` (initially the top interval).
    """
    p = np.asarray(p)
    pin = np.asarray(pin)
    ozmixt = np.asarray(ozmixt)
    if p.dtype != np.float32 or p.ndim != 2:
        raise ValueError("p must be float32 (ncol, nz)")
    if pin.dtype != np.float32 or pin.ndim != 1:
        raise ValueError("pin must be float32 (levsiz,)")
    levsiz = pin.size
    if not np.all(np.diff(pin) > 0):
        # WRF's runtime guard: 'OZN_P_INT: Bad ozone data: non-monotonicity
        # suspected' (wrf_error_fatal via the kount > ncol check).
        raise ValueError("OZN_P_INT: Bad ozone data: non-monotonicity "
                         "suspected")
    ncol, pver = p.shape
    if ozmixt.dtype != np.float32 or ozmixt.shape != (ncol, levsiz):
        raise ValueError("ozmixt must be float32 (ncol, levsiz)")

    pmid = p[:, ::-1]                       # top-down, as WRF reverses p
    kupper = np.zeros(ncol, dtype=np.int64)  # 0-based; WRF inits to 1
    cols = np.arange(ncol)
    o3vmr = np.empty((ncol, pver), dtype=np.float32)
    for k in range(pver):
        kout = pver - 1 - k                 # WRF: kout = pver - k + 1
        pm = pmid[:, k]
        # Bracket kk with pin(kk) < pm <= pin(kk+1) (unique; none when pm
        # <= pin(1) or pm > pin(levsiz)).
        cand = np.searchsorted(pin, pm, side="left").astype(np.int64) - 1
        found = (cand >= 0) & (cand <= levsiz - 2)
        kupper = np.where(found, cand, kupper)
        res = np.empty(ncol, dtype=np.float32)
        above = ~found & (pm < pin[0])
        below = ~found & (pm > pin[levsiz - 1])
        other = ~(above | below)            # bracketed, or pm == pin(1)
        if above.any():
            # o3vmr = ozmixt(i,1)*pmid/pin(1)
            res[above] = (ozmixt[above, 0] * pm[above]) / pin[0]
        if below.any():
            res[below] = ozmixt[below, levsiz - 1]
        if other.any():
            ku = kupper[other]
            dpu = pm[other] - pin[ku]
            dpl = pin[ku + 1] - pm[other]
            res[other] = (ozmixt[cols[other], ku] * dpl +
                          ozmixt[cols[other], ku + 1] * dpu) / (dpl + dpu)
        o3vmr[:, kout] = res
    return o3vmr


def o33d_profile(julday, julian, xlat, p, *,
                 data_dir: str | Path | None = None) -> np.ndarray:
    """O33D for a batch of columns, matching the driver's o3input=2 path.

    Replicates the exact call sequence WRF performs for the RRTMG
    option-4 schemes: oznini's latitude interpolation (init time),
    then ozn_time_int and ozn_p_int (radiation call time).

    julday : int; forwarded to ozn_time_int, which ignores it (as WRF
             does).
    julian : REAL day-of-year, 0.0 at 0Z on 1 Jan.
    xlat   : (ncol,) latitudes in degrees.
    p      : (ncol, nz) float32 layer-midpoint pressures, Pa, bottom-up
             (the driver's P3D tile).

    Returns O33D float32 (ncol, nz), bottom-up -- the values RRTMG_LWRAD/
    RRTMG_SWRAD consume directly as o3vmr for k <= kte.
    """
    xlat = np.asarray(xlat, dtype=np.float32)
    if xlat.ndim != 1:
        raise ValueError("xlat must be one-dimensional (ncol,)")
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[0] != xlat.shape[0]:
        raise ValueError("p must be (ncol, nz) with ncol == xlat.size")
    # Audit hardening: ozn_p_int's kupper walk assumes each column is
    # monotone (bottom-up decreasing); a non-monotone tile would walk a
    # stale kupper silently.  The radiation driver feeds the hydrostatic
    # p_hyd, monotone by construction, so this can only trip on a
    # miswired caller -- fail closed rather than interpolate garbage.
    if not bool(np.all(p[:, 1:] < p[:, :-1])):
        raise ValueError(
            "o33d_profile: p must be strictly decreasing bottom-up per "
            "column (the driver's hydrostatic P3D tile is); a "
            "non-monotone column would silently corrupt the ozn_p_int "
            "kupper walk")
    climo = load_ozone_climatology(data_dir)
    ozmixm = interp_ozone_to_latitudes(xlat, climo)
    ozmixt = ozn_time_int(julday, julian, ozmixm)
    return ozn_p_int(p, climo.plev, ozmixt)
