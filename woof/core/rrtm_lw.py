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
"""WRF v4.6.1 RRTM longwave radiation (``ra_lw_physics=1``).

Transcription of ``phys/module_ra_rrtm.F``'s forecast path: ``O3DATA``
(:3556-3667), ``MM5ATM`` (:3670-4180), ``SETCOEF`` (:4183-4366),
``GASABS``'s lookup-index tail (:6746-6754), ``RTRN`` (:6195-6600) and
the ``RRTM``/``RRTMLWRAD`` drivers (:1731-2298).  The band optical depths
live in :mod:`woof.core.rrtm_taumol`; the coefficient tables in
:mod:`woof.core.rrtm_tables`.

Column layout follows Dudhia's adapter: woof state is bottom-to-top,
WRF's 1-D radiation arrays are top-to-bottom, and
:func:`rrtmlwrad_columns` -- RRTMLWRAD's own packing, the seam the WRF
column oracle grades (tools/rrtm_lw_wrf461_oracle,
tests/test_rrtm_lw_wrf461_parity.py) -- owns that single packing boundary.  Inside, RRTM's own bottom-to-top ``1..NLAYERS``
indexing is used, extended above the model top by the Cavallo buffer
layers (``NLAYERS = kme + NINT(p_top_hPa/4) - 1``, ``rrtminit`` F:6783).

Documented divergences from a literal transliteration, each of which
guards a Fortran out-of-bounds read rather than changing a defined
result:

* ``RTRN``'s Planck-table subscripts ``INDBOUND``/``INDLEV``/``INDLAY``
  (F:6334, 6342, 6349) index ``TOTPLNK(181,...)`` directly from
  ``T - 159.``  Any level colder than 160 K or warmer than 340 K reads
  outside the table.  WRF guards only the surface value (``TBOUND`` is
  clamped to 339.99, F:3949-3953); level and layer temperatures are
  unguarded.  Here every one of the three is clamped into the 1-based
  range ``[1, 181]`` before :func:`_planck_index` converts it to the
  zero-based row, and on the last row the upper neighbour is the row
  itself, so the interpolation always lands inside the 181 rows for any
  finite temperature.  At exactly 340 K (a legal TBOUND) that is WRF's
  own value; above it WRF reads out of bounds and this port holds the
  340 K row.
* ``SETCOEF``'s round-off repair ``IF (JP(LAYSWTCH+1) .LE. 6)``
  (F:4363) reads one past the top layer when ``LAYSWTCH == NLAYERS``.
  The index is clamped to the last layer.
* ``RTRN``'s cloudy-layer scratch (``BBUTOT``, ``ODCLR``, ``ODCLD``,
  ``ABSCLD``, ``EFCLFRAC``) is declared ``kts:ktep1-1`` but written at
  ``LEV`` up to ``NLAYERS`` (F:6323-6324, 6371).  It is unreachable
  because MM5ATM zeroes ``CLDFRAC`` above the model top, so the top
  layer is never cloudy; the arrays here are simply full length.

Nothing else is smoothed over.  In particular the port keeps WRF's
second ozone interpolation off by one layer relative to the first: the
buffer-zone call reverses ``PZ`` such that ``PRLEVH(K) == PZ(K)``
(F:4106-4110) where the model-column call gives ``PRLEVH(K) == PZ(K-1)``.
That is what WRF computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from woof.core.rrtm import (RRTM_DELTAP_HPA, rrtm_default_trace_gases,
                             rrtm_upper_layer_count)
from woof.core.rrtm_tables import (RRTM_BPADE, RRTM_FLUXFAC, RRTM_NBANDS,
                                    RRTM_SECANG, RRTM_WTNUM,
                                    load_rrtm_lw_tables)
from woof.core.rrtm_taumol import RRTM_NGPT, _array_namespace, rrtm_taumol


#: MM5ATM's molecular-weight ratios and cloud absorption coefficients,
#: module_ra_rrtm.F:3838-3890.
_AMD = 28.9644
_AMW = 18.0154
_AMDW = 1.607758
_AMDO = 0.603461
_AMDC = 1.805423
_AMDN = 0.658090
_AVGDRO = 6.022e23
_H2OVMR = 5.00e-6
_ABCW = 0.144
_ABICE = 0.0735
_ABRN = 0.330e-3
_ABSN = 2.34e-3
#: RRTM's own upper bound on the Planck lookup, F:3949-3953.
_TBOUND_MAX = 339.99
#: TOTPLNK's first dimension (module_ra_rrtm.F:110).
_TOTPLNK_ROWS = 181
#: Cold CUDA-graph capture fallback for low-level automatic callers. The
#: eager auto-sizer may choose fewer columns when its allocation allowance
#: requires it. Ordinary experiments use their explicit common column cap.
#: Chunking changes dispatch/memory, not per-column radiation arithmetic.
DEFAULT_COLUMN_CHUNK = 512
#: Fraction of the currently reusable device memory (runtime free plus
#: the CuPy pool's free blocks) the auto-sized transfer may claim.
_TRANSIENT_MEMORY_FRACTION = 0.5
#: Allocation shapes, shared with admission; no fitted per-layer constant.
from woof.core.rrtm_inventory import auto_column_chunk, effective_column_chunk


def _resolve_column_chunk(configured, ncol: int, nlayers: int, xp) -> int:
    """Columns per solver pass: the configured pin, or the memory fit.

    ``None`` sizes the chunk so the solve's transient arrays fit inside
    ``_TRANSIENT_MEMORY_FRACTION`` of what the device can hand back
    without growing its footprint, capped at the block's own column count.
    A smaller block is allowed whenever the former 512-column floor would
    exceed this allowance.  Host (NumPy) blocks
    have no device pool to protect and run in one pass.

    NEVER CALL THIS INSIDE CUDA GRAPH CAPTURE: ``cudaMemGetInfo`` is not
    a capturable operation and the runtime refuses it mid-capture.  The
    adapters resolve the auto-size at their first EAGER solve and cache
    it (:meth:`RRTMLongwaveRadiation._column_chunk_for`); a capture that
    arrives cold uses the capture fallback, byte identical by the
    chunk-invariance guarantee.
    """
    if configured is not None:
        return int(configured)
    cuda = getattr(xp, "cuda", None)
    if cuda is None:
        return max(int(ncol), 1)
    free_device, _total = cuda.runtime.memGetInfo()
    reusable = int(free_device) + int(xp.get_default_memory_pool().free_bytes())
    return auto_column_chunk(ncol, nlayers,
                             int(reusable * _TRANSIENT_MEMORY_FRACTION))


def _stream_is_capturing(cp) -> bool:
    """True when the current stream is inside CUDA graph capture.

    Defensive on purpose: an ``is_capturing`` probe that itself raised
    (an exotic stream object, an old CuPy) must degrade to the eager
    answer rather than take the radiation step down, so any failure here
    reads as "not capturing" and the eager path runs exactly as before.
    """
    try:
        return bool(cp.cuda.get_current_stream().is_capturing())
    except Exception:                                   # pragma: no cover
        return False


#: Constant coefficient tables moved onto the array namespace once per
#: process, keyed by (table id, builder name, namespace, dtype).  Before
#: this cache every solver chunk re-uploaded its tables from pageable
#: host memory -- hundreds of synchronous H2D copies per radiation step.
_DEVICE_CONSTANT_CACHE: dict[tuple, object] = {}


def _device_constant(xp, dtype, key, build):
    """Return ``xp.asarray(build(), dtype)`` cached per process."""
    # PER CARD for CuPy: a table uploaded on one card is not readable on
    # another, and the [devices] split steps slabs on several cards in one
    # process.  The host (NumPy) key is unchanged.
    device = (int(xp.cuda.Device().id) if xp.__name__ == "cupy" else None)
    cache_key = (*key, xp.__name__, np.dtype(dtype).str, device)
    if device is not None:
        from woof.core.device_cache import cached_ready
        return cached_ready(xp, _DEVICE_CONSTANT_CACHE, cache_key,
                            lambda: xp.asarray(build(), dtype=dtype))
    value = _DEVICE_CONSTANT_CACHE.get(cache_key)
    if value is None:
        value = xp.asarray(build(), dtype=dtype)
        _DEVICE_CONSTANT_CACHE[cache_key] = value
    return value


# ---------------------------------------------------------------------------
# The float32 words gfortran's REAL arithmetic resolves to.
#
# WRF's LOG, EXP and ``**`` on REAL are the libm logf/expf/powf, which are
# neither CUDA's device functions nor NumPy's float32 loops.  WOOF carries
# its own float32 libm: on the host woof.core.noahmp_libm, on the device
# the elementwise slab kernels of woof.core.noahmp_slab_libm, held bitwise
# to the host one by tests/test_noahmp_slab_libm.py.  RRTM uses no other
# transcendental.  The WRF column oracle (tools/rrtm_lw_wrf461_oracle)
# measured CuPy's builtins against them here: PLOG, CO2MULT and the
# rain/snow paths each moved by ULPs, which the band optical depths then
# amplified.
# ---------------------------------------------------------------------------

def _logf(xp, x):
    if xp is np:
        from woof.core.noahmp_libm import logf_array
        return logf_array(x)
    from woof.core.noahmp_slab_libm import slab_logf
    return slab_logf(x)


def _expf(xp, x):
    if xp is np:
        from woof.core.noahmp_libm import expf
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        return np.fromiter((expf(v) for v in flat), dtype=np.float32,
                           count=flat.size).reshape(np.shape(x))
    from woof.core.noahmp_slab_libm import slab_expf
    return slab_expf(x)


def _powf(xp, x, exponent):
    if xp is np:
        from woof.core.noahmp_libm import powf_array
        return powf_array(x, np.float32(exponent))
    from woof.core.noahmp_slab_libm import slab_powf
    return slab_powf(x, exponent)


_SEQUENTIAL_SUM = None


def _sequential_sum(xp, values):
    """Sum over the last axis in index order, from +0.0, one rounding per add.

    Every sum RRTM forms is a Fortran accumulation loop (O3DATA's
    ``O3PROF(K)=O3PROF(K)+...`` over 31 climatology layers, RTRN's
    ``DRAD = DRAD + RADLD(IPR)`` over 140 g-points), so its rounding
    depends on the ORDER.  ``xp.sum`` is pairwise on NumPy and a tree
    reduction on CuPy; neither is that order.  The oracle measured the
    tree sum up to ~4e4 ULP off O3PROF.
    """
    if xp is np:
        values = np.asarray(values)
        total = np.zeros(values.shape[:-1], dtype=values.dtype)
        for index in range(values.shape[-1]):
            total = total + values[..., index]
        return total
    global _SEQUENTIAL_SUM
    if _SEQUENTIAL_SUM is None:
        # One thread per row.  Additions only, so there is nothing for
        # NVRTC to contract into an FMA.
        _SEQUENTIAL_SUM = xp.ElementwiseKernel(
            "raw T x, int32 n", "T y",
            "T acc = 0; const long long base = (long long)i * n; "
            "for (int j = 0; j < n; ++j) { acc = acc + x[base + j]; } "
            "y = acc;",
            "gpuwm_rrtm_sequential_sum")
    values = xp.ascontiguousarray(values)
    out = xp.empty(values.shape[:-1], dtype=values.dtype)
    if out.size:
        _SEQUENTIAL_SUM(values, np.int32(values.shape[-1]), out)
    return out


def rrtm_layer_count(nz: int, p_top_pa: float) -> int:
    """``NLAYERS`` for a column of ``nz`` mass levels (``rrtminit`` F:6783).

    WRF computes ``kme + NINT(p_top*0.01/deltap) - 1`` with ``kme = kte+1``
    and ``kte = nz``, i.e. the model layers plus the Cavallo buffer.
    """
    if nz < 3:
        raise ValueError("RRTM longwave needs at least three model layers")
    return int(nz) + rrtm_upper_layer_count(p_top_pa)


@lru_cache(maxsize=1)
def _o3_climatology() -> tuple[np.ndarray, np.ndarray]:
    """``O3WRK``/``PPWRKH`` from O3DATA's summer/winter blend (F:3634-3660)."""
    from woof.core.rrtm_tables import load_rrtm_lw_statics

    statics = load_rrtm_lw_statics()
    o3sum = statics["O3DATA__O3SUM"].astype(np.float32)
    o3win = statics["O3DATA__O3WIN"].astype(np.float32)
    ppsum = statics["O3DATA__PPSUM"].astype(np.float32)
    ppwin = statics["O3DATA__PPWIN"].astype(np.float32)
    ppann = ppsum.copy()
    o3ann = np.empty(31, dtype=np.float32)
    o3ann[0] = np.float32(0.5) * (o3sum[0] + o3win[0])
    for k in range(1, 31):
        o3ann[k] = (o3win[k - 1]
                    + (o3win[k] - o3win[k - 1]) / (ppwin[k] - ppwin[k - 1])
                    * (ppsum[k] - ppwin[k - 1]))
    for k in range(1, 31):
        o3ann[k] = np.float32(0.5) * (o3ann[k] + o3sum[k])
    ppwrkh = np.empty(32, dtype=np.float32)
    ppwrkh[0] = np.float32(1100.0)
    for k in range(1, 31):
        ppwrkh[k] = (ppann[k] + ppann[k - 1]) / np.float32(2.0)
    ppwrkh[31] = np.float32(0.0)
    o3ann.setflags(write=False)
    ppwrkh.setflags(write=False)
    return o3ann, ppwrkh


def o3data_columns(xp, prlevh, nlay: int):
    """``O3DATA``'s pressure-weighted layer ozone (F:3646-3665).

    ``prlevh`` is ``(ncol, nlay+1)`` bottom-to-top level pressure in mb,
    matching the Fortran's reversed ``PRLEVH``.  Returns ``(ncol, nlay)``
    mass mixing ratio.
    """
    dtype = prlevh.dtype
    o3wrk = _device_constant(
        xp, dtype, ("o3data", "o3wrk"), lambda: _o3_climatology()[0])
    ppwrkh = _device_constant(
        xp, dtype, ("o3data", "ppwrkh"), lambda: _o3_climatology()[1])
    bottom = prlevh[:, :nlay, None]
    top = prlevh[:, 1:nlay + 1, None]
    zero = dtype.type(0.0)
    # PB1/PB2/PT1/PT2: the Fortran's "IF (-(P-PH)) >= 0 THEN 0 ELSE P-PH"
    # is exactly a positive part.
    pb1 = xp.maximum(bottom - ppwrkh[None, None, :31], zero)
    pb2 = xp.maximum(bottom - ppwrkh[None, None, 1:32], zero)
    pt1 = xp.maximum(top - ppwrkh[None, None, :31], zero)
    pt2 = xp.maximum(top - ppwrkh[None, None, 1:32], zero)
    weighted = _sequential_sum(xp, (pb2 - pb1 - pt2 + pt1)
                               * o3wrk[None, None, :])
    return weighted / (prlevh[:, :nlay] - prlevh[:, 1:nlay + 1])


def _standard_atmosphere_temperature(xp, statics, pz):
    """Interpolate MM5ATM's mean ``TPROF`` to ``PZ`` (F:4069-4093)."""
    dtype = pz.dtype
    pprof = _device_constant(
        xp, dtype, (id(statics), "MM5ATM__PPROF"),
        lambda: statics["MM5ATM__PPROF"])
    tprof = _device_constant(
        xp, dtype, (id(statics), "MM5ATM__TPROF"),
        lambda: statics["MM5ATM__TPROF"])
    nprof = int(pprof.shape[0])
    # klev = LL-1 for the first LL>=2 with PPROF(LL) < PZ; PPROF is
    # monotonically decreasing, so that is the count of entries at or
    # above PZ, floored at 1.  The outer guard sends PZ <= PPROF(last)
    # straight to the last level.
    above = (pprof[None, None, :] >= pz[..., None]).sum(axis=-1)
    klev = xp.clip(above, 1, nprof)
    klev = xp.where(pz > pprof[nprof - 1], klev, nprof).astype(xp.int32)
    last = klev == nprof
    k0 = xp.clip(klev - 1, 0, nprof - 1)
    k1 = xp.clip(klev, 0, nprof - 1)
    vark = tprof[k0]
    vark1 = xp.where(last, vark, tprof[k1])
    # klev == nprof collapses k0 and k1 onto the same table entry, so the
    # Fortran's ELSE branch sets WGHT = 0 without dividing.  Neutralise
    # the denominator rather than divide and discard.
    span = pprof[k1] - pprof[k0]
    span = xp.where(last, dtype.type(1.0), span)
    wght = xp.where(last, dtype.type(0.0), (pz - pprof[k0]) / span)
    return wght * (vark1 - vark) + vark


def mm5atm_columns(xp, tables, *, t, p_mb, dz, qv, qc, qr, qi, qs, qg,
                   cldfra, tw, pw_mb, tsfc, emiss, r, g,
                   co2vmr, n2ovmr, ch4vmr, nlayers):
    """``MM5ATM``: build RRTM's profile arrays (F:3670-4180).

    Layer inputs are ``(ncol, nz)`` TOP-DOWN; level inputs ``(ncol, nz+1)``
    TOP-DOWN; ``tsfc``/``emiss`` are ``(ncol,)``.  Everything returned is
    RRTM's bottom-to-top ordering: layer arrays ``(ncol, nlayers)`` and
    level arrays ``(ncol, nlayers+1)`` indexed ``PZ(0..NLAYERS)``.
    """
    dtype = t.dtype
    real = dtype.type
    ncol, nz = t.shape
    gravit = real(g) * real(100.0)

    #  MID-LAYER VALUES, F:3891-3915.
    qv = xp.maximum(qv, real(1.0e-12))
    ro = p_mb / (real(r) * t) * real(100.0)
    clwp = ro * qc * dz * real(1000.0)
    ciwp = ro * qi * dz * real(1000.0)
    plwp = _powf(xp, ro * qr, real(0.75)) * dz * real(1000.0)
    piwp = _powf(xp, ro * qs, real(0.75)) * dz * real(1000.0)

    #  TBOUND, F:3946-3953.  WRF resets a hotter surface to 339.99 and
    #  prints; here it is a silent clamp because there is no per-column
    #  message channel.
    tbound = xp.where(tsfc > real(340.0), real(_TBOUND_MAX), tsfc)

    wkl = xp.zeros((ncol, nlayers, 35), dtype=dtype)
    wx = xp.zeros((ncol, nlayers, 4), dtype=dtype)
    pavel = xp.zeros((ncol, nlayers), dtype=dtype)
    tavel = xp.zeros((ncol, nlayers), dtype=dtype)
    coldry = xp.zeros((ncol, nlayers), dtype=dtype)
    pz = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    tz = xp.zeros((ncol, nlayers + 1), dtype=dtype)

    # F:3956-3979.  PAVEL(L)=p(kte+1-L) etc: a straight reversal.
    pavel[:, :nz] = p_mb[:, ::-1]
    tavel[:, :nz] = t[:, ::-1]
    pz[:, :nz + 1] = pw_mb[:, ::-1]
    tz[:, :nz + 1] = tw[:, ::-1]

    #  Ozone on the model column.  INIRAD/O3DATA take PRLEVH(K)=PZ(K-1),
    #  so the model-layer call sees each layer's own bounding levels.
    o3prof = o3data_columns(xp, pz[:, :nz + 1], nz)

    wkl[:, :nz, 0] = qv[:, ::-1] * real(_AMDW)
    wkl[:, :nz, 1] = real(co2vmr)
    # F:3971-3972: WKL(3,L)=o3(kte+1-L) is overwritten on the next line
    # by WKL(3,L)=o3(L)*amdo.  The first assignment is dead in WRF; the
    # ozone profile is already bottom-to-top.
    wkl[:, :nz, 2] = o3prof * real(_AMDO)
    # N2O/CH4 are filled as constants by F:3917-3918 (n2ovmr/amdn and
    # ch4vmr/amdc) and multiplied straight back by amdn/amdc here.
    # N2O(:) = n2ovmr / amdn divides the REAL(8) gas by the REAL amdn in
    # binary64 and stores into REAL; WKL then multiplies back in float32.
    wkl[:, :nz, 3] = (real(float(n2ovmr) / float(real(_AMDN)))
                      * real(_AMDN))
    wkl[:, :nz, 5] = (real(float(ch4vmr) / float(real(_AMDC)))
                      * real(_AMDC))
    amm = ((real(1.0) - wkl[:, :nz, 0]) * real(_AMD)
           + wkl[:, :nz, 0] * real(_AMW))
    coldry[:, :nz] = ((pz[:, :nz] - pz[:, 1:nz + 1]) * real(1.0e3)
                      * real(_AVGDRO)
                      / (gravit * amm * (real(1.0) + wkl[:, :nz, 0])))

    #  Cavallo buffer levels, F:4030-4041.
    deltap = real(RRTM_DELTAP_HPA)
    for level in range(nz + 1, nlayers):
        pz[:, level] = pz[:, level - 1] - deltap
        pavel[:, level - 1] = real(0.5) * (pz[:, level] + pz[:, level - 1])
    pz[:, nlayers] = real(0.0)
    pavel[:, nlayers - 1] = real(0.5) * (pz[:, nlayers] + pz[:, nlayers - 1])

    #  Buffer temperatures from the mean standard atmosphere, F:4043-4073.
    varint = _standard_atmosphere_temperature(
        xp, tables.statics, pz[:, 1:nlayers + 1])
    if nlayers > nz:
        offset = (tz[:, nz] - varint[:, nz - 1])[:, None]
        tz[:, nz + 1:nlayers + 1] = varint[:, nz:nlayers] + offset
        for level in range(nz + 1, nlayers + 1):
            tavel[:, level - 1] = real(0.5) * (tz[:, level] + tz[:, level - 1])

    #  Buffer-zone ozone, F:4075-4103.  PZR reverses PZ(1..NLAYERS), so
    #  O3DATA's PRLEVH(K) lands on PZ(K) -- one layer up from the model
    #  column's PZ(K-1).  Reproduced as WRF computes it.
    o3prof2 = xp.empty((ncol, nlayers), dtype=dtype)
    o3prof2[:, :nlayers - 1] = o3data_columns(
        xp, pz[:, 1:nlayers + 1], nlayers - 1)
    o3prof2[:, nlayers - 1] = real(6.135e-6)
    if nz != nlayers:
        wkl[:, :, 2] = o3prof2 * real(_AMDO)
        wkl[:, nz:, 0] = real(_H2OVMR)
        wkl[:, nz:, 1] = real(co2vmr)
        wkl[:, nz:, 3] = wkl[:, nz - 1:nz, 3]
        wkl[:, nz:, 5] = wkl[:, nz - 1:nz, 5]
        amm_buf = ((real(1.0) - wkl[:, nz:, 0]) * real(_AMD)
                   + wkl[:, nz:, 0] * real(_AMW))
        coldry[:, nz:] = ((pz[:, nz:nlayers] - pz[:, nz + 1:nlayers + 1])
                          * real(1.0e3) * real(_AVGDRO)
                          / (gravit * amm_buf
                             * (real(1.0) + wkl[:, nz:, 0])))
        wx[:, nz:, 1] = wx[:, nz - 1:nz, 1]
        wx[:, nz:, 2] = wx[:, nz - 1:nz, 2]

    #  vmr -> molec/cm2, F:4105-4114.  NMOL=6 and the IXINDX cross-section
    #  remap (IXINDX = 0,2,3,0) both come from the packaged statics.
    wkl[:, :, :6] = wkl[:, :, :6] * coldry[:, :, None]
    ixindx = tables.statics["IXINDX"]
    for ix in range(4):
        target = int(ixindx[ix])
        if target != 0:
            wx[:, :, target - 1] = (coldry * wx[:, :, ix]
                                    * real(1.0e-20))

    #  Spectral surface emissivity, F:4121-4123: one land-use value for
    #  every band.
    semiss = xp.broadcast_to(emiss[:, None], (ncol, RRTM_NBANDS))

    #  Cloud optical depth and fraction, F:4133-4145.
    taucloud = xp.zeros((ncol, nlayers), dtype=dtype)
    tau_model = (real(_ABCW) * clwp[:, ::-1] + real(_ABICE) * ciwp[:, ::-1]
                 + real(_ABRN) * plwp[:, ::-1] + real(_ABSN) * piwp[:, ::-1])
    taucloud[:, :nz] = tau_model
    cldfrac = xp.zeros((ncol, nlayers), dtype=dtype)
    cldfrac[:, :nz] = xp.where(tau_model > real(0.01), real(1.0),
                               cldfra[:, ::-1])
    return {
        "pavel": pavel, "tavel": tavel, "pz": pz, "tz": tz,
        "cldfrac": cldfrac, "taucloud": taucloud, "coldry": coldry,
        "wkl": wkl, "wx": wx, "tbound": tbound, "semiss": semiss,
        "o3prof": o3prof,
    }


def setcoef_columns(xp, tables, *, pavel, tavel, coldry, wkl):
    """``SETCOEF``: interpolation indices and column amounts (F:4183-4366)."""
    dtype = pavel.dtype
    real = dtype.type
    ncol, nlayers = pavel.shape
    preflog = _device_constant(
        xp, dtype, (id(tables), "PREFLOG"),
        lambda: tables.statics["PREFLOG"])
    tref = _device_constant(
        xp, dtype, (id(tables), "TREF"), lambda: tables.statics["TREF"])
    # STPFAC = 296./1013.: two REAL literals, a float32 quotient.
    stpfac = real(296.0) / real(1013.0)

    plog = _logf(xp, pavel)
    jp = xp.clip(xp.trunc(real(36.0) - real(5.0) * (plog + real(0.04))
                          ).astype(xp.int32), 1, 58)
    fp = real(5.0) * (preflog[jp - 1] - plog)
    jt = xp.clip(xp.trunc(real(3.0) + (tavel - tref[jp - 1]) / real(15.0)
                          ).astype(xp.int32), 1, 4)
    ft = ((tavel - tref[jp - 1]) / real(15.0)) - (jt - 3).astype(dtype)
    jt1 = xp.clip(xp.trunc(real(3.0) + (tavel - tref[jp]) / real(15.0)
                           ).astype(xp.int32), 1, 4)
    ft1 = ((tavel - tref[jp]) / real(15.0)) - (jt1 - 3).astype(dtype)

    water = wkl[:, :, 0] / coldry
    scalefac = pavel * stpfac / tavel

    #  LAYTROP/LAYSWTCH/LAYLOW are COUNTS over the whole column
    #  (F:4291-4295), which the branch bounds "DO LAY = 1, LAYTROP" then
    #  treat as a contiguous split.
    troposphere = plog > real(4.56)
    laytrop = troposphere.sum(axis=1).astype(xp.int32)
    layswtch = (troposphere & (plog > real(5.76))).sum(axis=1).astype(xp.int32)
    laylow = (troposphere & (plog >= real(6.62))).sum(axis=1).astype(xp.int32)

    forfac = scalefac / (real(1.0) + water)
    selffac = water * forfac
    factor = (tavel - real(188.0)) / real(7.2)
    indself = xp.clip(xp.trunc(factor).astype(xp.int32) - 7, 1, 9)
    selffrac = factor - (indself + 7).astype(dtype)
    #  Above LAYTROP the self-continuum terms are not evaluated at all
    #  (F:4327-4344 omits them), so their values there are irrelevant;
    #  TAUMOL's masks discard them.

    scale = real(1.0e-20)
    colh2o = scale * wkl[:, :, 0]
    colco2 = scale * wkl[:, :, 1]
    colo3 = scale * wkl[:, :, 2]
    coln2o = scale * wkl[:, :, 3]
    colch4 = scale * wkl[:, :, 5]
    colo2 = scale * wkl[:, :, 6]
    floor = real(1.0e-32) * coldry
    colco2 = xp.where(colco2 == real(0.0), floor, colco2)
    coln2o = xp.where(coln2o == real(0.0), floor, coln2o)
    colch4 = xp.where(colch4 == real(0.0), floor, colch4)
    co2reg = real(3.55e-24) * coldry
    co2mult = ((colco2 - co2reg) * real(272.63)
               * _expf(xp, real(-1919.4) / tavel)
               / (real(8.7604e-4) * tavel))

    compfp = real(1.0) - fp
    fac10 = compfp * ft
    fac00 = compfp * (real(1.0) - ft)
    fac11 = fp * ft1
    fac01 = fp * (real(1.0) - ft1)

    #  F:4361-4365.  LAYLOW=0 becomes 1, and a round-off repair bumps
    #  LAYSWTCH when the next layer is still below the ~300 mb switch.
    #  JP(LAYSWTCH+1) reads one past the top when LAYSWTCH == NLAYERS;
    #  the index is clamped (see the module docstring).
    laylow = xp.where(laylow == 0, 1, laylow)
    probe = xp.clip(layswtch, 0, nlayers - 1)
    jp_next = xp.take_along_axis(jp, probe[:, None], axis=1)[:, 0]
    layswtch = xp.where(jp_next <= 6, layswtch + 1, layswtch)

    return {
        "colh2o": colh2o, "colco2": colco2, "colo3": colo3,
        "coln2o": coln2o, "colch4": colch4, "colo2": colo2,
        "co2mult": co2mult, "fac00": fac00, "fac01": fac01,
        "fac10": fac10, "fac11": fac11, "forfac": forfac,
        "selffac": selffac, "selffrac": selffrac, "jp": jp, "jt": jt,
        "jt1": jt1, "indself": indself, "laytrop": laytrop,
        "layswtch": layswtch, "laylow": laylow,
    }


def _planck_index(xp, temperature, dtype):
    """``INT(T - 159.)`` and its fraction, as a ZERO-BASED TOTPLNK row.

    Fortran's ``INDBOUND = INT(T - 159.)`` (F:6334, 6342, 6349) is a
    1-based subscript into ``TOTPLNK(181,16)``, whose first row is 160 K,
    so ``TOTPLNK(i)`` is the row for ``159 + i`` kelvin.  The zero-based
    row for the same temperature is ``i - 1``, and that subtraction is
    the whole difference between this function and the Fortran: without
    it every Planck evaluation reads one kelvin too warm (about +1.3% on
    GLW at 288 K) and a clamped 339.99 K surface runs off the end of the
    array.  The sibling ``TAU``/``TF``/``TRANS`` tables are declared
    ``0:5000`` in the Fortran and are indexed here without any shift;
    the two conventions really do sit side by side in the source.

    The clamp is applied to the 1-based value and keeps the row inside
    ``[0, 180]``, the LAST row included: 340.0 K exactly is a legal TBOUND
    (MM5ATM resets only ``TBOUND .GT. 340.``), it indexes row 181 with a
    zero fraction, and WRF's value there is ``TOTPLNK(181)`` itself.  The
    old clamp to row 180 read the 339 K row instead -- 2.3 W m-2 of OLR on
    the oracle's 340 K surface.  See the module docstring for why a clamp
    exists at all.
    """
    index = xp.trunc(temperature - dtype.type(159.0)).astype(xp.int32)
    frac = temperature - xp.trunc(temperature)
    return xp.clip(index, 1, _TOTPLNK_ROWS) - 1, frac


def _planck_bands(totplnk, delwave, row, frac):
    """``DELWAVE*(TOTPLNK(row) + frac*(TOTPLNK(row+1)-TOTPLNK(row)))``.

    F:6371-6395, with ``row`` the zero-based row from
    :func:`_planck_index`.  Hoisted out of :func:`rtrn_columns` so the
    Stefan-Boltzmann table check in the test suite exercises the shipped
    arithmetic rather than a copy of it.

    On the last row the upper neighbour is the row itself.  At exactly
    340 K the fraction is zero and the result is ``TOTPLNK(181)``, which
    is what WRF computes (its ``TOTPLNK(182)`` operand is multiplied by
    that zero).  Above 340 K WRF reads past the table (a level or layer
    temperature is never clamped there); this port holds the last row
    instead, a divergence documented in the module docstring.
    """
    low = totplnk[row]
    high = totplnk[(row + 1).clip(0, _TOTPLNK_ROWS - 1)]
    return delwave * (low + frac[..., None] * (high - low))


def rtrn_columns(xp, tables, *, tavel, pz, tz, cldfrac, taucloud, itr,
                 pfrac, tbound, semiss):
    """``RTRN``: the two-stream diffusivity-angle transfer (F:6195-6600).

    Returns ``(totdflux, totuflux, htr, htrc)`` with level arrays
    ``(ncol, nlayers+1)`` and heating rates in K/day.
    """
    dtype = tavel.dtype
    real = dtype.type
    ncol, nlayers = tavel.shape
    totplnk = _device_constant(
        xp, dtype, (id(tables), "TOTPLNK"),
        lambda: tables.statics["TOTPLNK"].reshape(
            (_TOTPLNK_ROWS, RRTM_NBANDS), order="F"))
    delwave = _device_constant(
        xp, dtype, (id(tables), "DELWAVE"),
        lambda: tables.statics["DELWAVE"])
    heatfac = real(tables.statics["HEATFAC"][0])
    ngb0 = _device_constant(
        xp, np.int32, (id(tables), "NGB0"),
        lambda: np.asarray(tables.statics["NGB"], dtype=np.int32) - 1)
    tau_tab = _device_constant(
        xp, dtype, (id(tables), "TAU"), lambda: tables.tau)
    tf_tab = _device_constant(
        xp, dtype, (id(tables), "TF"), lambda: tables.tf)
    trans_tab = _device_constant(
        xp, dtype, (id(tables), "TRANS"), lambda: tables.trans)

    indbound, tbndfrac = _planck_index(xp, tbound, dtype)
    indlev, tlevfrac = _planck_index(xp, tz, dtype)
    indlay, tlayfrac = _planck_index(xp, tavel, dtype)

    #  Integrated Planck functions, F:6371-6395.
    def planck(row, frac):
        return _planck_bands(totplnk, delwave, row, frac)

    plankbnd = planck(indbound, tbndfrac)          # (ncol, nbands)
    plvl = planck(indlev, tlevfrac)                # (ncol, nlayers+1, nbands)
    play = planck(indlay, tlayfrac)                # (ncol, nlayers, nbands)
    plnkemit = semiss * plankbnd

    #  Cloud optics, F:6350-6360.
    odcld = RRTM_SECANG.astype(dtype) * taucloud
    abscld = real(1.0) - _expf(xp, -odcld)
    efclfrac = abscld * cldfrac
    icldlyr = cldfrac > real(0.0)

    odclr = tau_tab[itr]
    tauf = tf_tab[itr]
    abss = real(1.0) - trans_tab[itr]

    semis = semiss[:, ngb0]
    raduemit = pfrac[:, 0, :] * plnkemit[:, ngb0]
    bglev = pfrac[:, nlayers - 1, :] * plvl[:, nlayers, :][:, ngb0]

    radld = xp.zeros((ncol, RRTM_NGPT), dtype=dtype)
    radclrd = xp.zeros((ncol, RRTM_NGPT), dtype=dtype)
    totdflux = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    totdclfl = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    bbu = xp.zeros((ncol, nlayers, RRTM_NGPT), dtype=dtype)
    bbutot = xp.zeros((ncol, nlayers, RRTM_NGPT), dtype=dtype)
    atot = xp.zeros((ncol, nlayers, RRTM_NGPT), dtype=dtype)
    iclddn = xp.zeros((ncol,), dtype=bool)

    #  DOWNWARD RADIATIVE TRANSFER, F:6416-6499.
    for lev in range(nlayers, 0, -1):
        k = lev - 1
        cloudy = icldlyr[:, k]
        iclddn = iclddn | cloudy
        cloudy3 = cloudy[:, None]
        bglay = pfrac[:, k, :] * play[:, k, :][:, ngb0]
        delbgup = bglev - bglay
        layer_tauf = tauf[:, k, :]
        bbu[:, k, :] = bglay + layer_tauf * delbgup
        bglev = pfrac[:, k, :] * plvl[:, lev - 1, :][:, ngb0]
        delbgdn = bglev - bglay
        bbd = bglay + layer_tauf * delbgdn
        layer_abss = abss[:, k, :]

        odsm = odclr[:, k, :] + odcld[:, k, None]
        factot = odsm / (RRTM_BPADE.astype(dtype) + odsm)
        bbutot[:, k, :] = bglay + factot * delbgup
        bbdlevd = bglay + factot * delbgdn
        layer_atot = (layer_abss + abscld[:, k, None]
                      - layer_abss * abscld[:, k, None])
        atot[:, k, :] = layer_atot
        gassrc = bbd * layer_abss

        radld_cloudy = (radld - radld * (layer_abss + efclfrac[:, k, None]
                                         * (real(1.0) - layer_abss))
                        + gassrc + cldfrac[:, k, None]
                        * (bbdlevd * layer_atot - gassrc))
        radld_clear = radld + (bbd - radld) * layer_abss
        radclrd_step = radclrd + (bbd - radclrd) * layer_abss
        radld = xp.where(cloudy3, radld_cloudy, radld_clear)
        #  In a clear layer the clear stream only separates once a cloud
        #  has been crossed; until then it tracks the total stream
        #  (F:6485-6492).
        radclrd = xp.where(cloudy3 | iclddn[:, None], radclrd_step, radld)
        totdflux[:, lev - 1] = (_sequential_sum(xp, radld)
                                * RRTM_WTNUM.astype(dtype))
        totdclfl[:, lev - 1] = (_sequential_sum(xp, radclrd)
                                * RRTM_WTNUM.astype(dtype))

    #  Surface emission and reflection, F:6506-6520.
    radlu = raduemit + (real(1.0) - semis) * radld
    radclru = raduemit + (real(1.0) - semis) * radclrd
    totuflux = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    totuclfl = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    totuflux[:, 0] = _sequential_sum(xp, radlu) * RRTM_WTNUM.astype(dtype)
    totuclfl[:, 0] = _sequential_sum(xp, radclru) * RRTM_WTNUM.astype(dtype)

    #  UPWARD RADIATIVE TRANSFER, F:6527-6574.
    for lev in range(1, nlayers + 1):
        k = lev - 1
        cloudy3 = icldlyr[:, k][:, None]
        layer_abss = abss[:, k, :]
        gassrc = bbu[:, k, :] * layer_abss
        radlu_cloudy = (radlu - radlu * (layer_abss + efclfrac[:, k, None]
                                         * (real(1.0) - layer_abss))
                        + gassrc + cldfrac[:, k, None]
                        * (bbutot[:, k, :] * atot[:, k, :] - gassrc))
        radlu_clear = radlu + (bbu[:, k, :] - radlu) * layer_abss
        radlu = xp.where(cloudy3, radlu_cloudy, radlu_clear)
        radclru = radclru + (bbu[:, k, :] - radclru) * layer_abss
        totuflux[:, lev] = (_sequential_sum(xp, radlu)
                            * RRTM_WTNUM.astype(dtype))
        totuclfl[:, lev] = (_sequential_sum(xp, radclru)
                            * RRTM_WTNUM.astype(dtype))

    #  Radiances to fluxes and heating rates, F:6580-6596.
    fluxfac = RRTM_FLUXFAC.astype(dtype)
    totuflux = totuflux * fluxfac
    totdflux = totdflux * fluxfac
    totuclfl = totuclfl * fluxfac
    totdclfl = totdclfl * fluxfac
    fnet = totuflux - totdflux
    fnetc = totuclfl - totdclfl
    dp = pz[:, :nlayers] - pz[:, 1:nlayers + 1]
    htr = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    htrc = xp.zeros((ncol, nlayers + 1), dtype=dtype)
    htr[:, :nlayers] = heatfac * (fnet[:, :nlayers]
                                  - fnet[:, 1:nlayers + 1]) / dp
    htrc[:, :nlayers] = heatfac * (fnetc[:, :nlayers]
                                   - fnetc[:, 1:nlayers + 1]) / dp
    return totdflux, totuflux, htr, htrc


def rrtm_longwave_columns(*, t, p_mb, dz, qv, qc, qr, qi, qs, qg, cldfra,
                          tw, pw_mb, tsfc, emiss, r, g, co2vmr, n2ovmr,
                          ch4vmr, nlayers, tables=None, stages=False):
    """``RRTM``'s per-column driver (F:2059-2298) for a block of columns.

    Layer inputs are ``(ncol, nz)`` TOP-DOWN, level inputs
    ``(ncol, nz+1)`` TOP-DOWN, pressures in mb.  Returns a dict with
    ``tten``/``ttenc`` (``(ncol, nz)`` BOTTOM-UP temperature tendency,
    K s-1), ``glw`` and ``olr`` (``(ncol,)``, W m-2).
    """
    if tables is None:
        tables = load_rrtm_lw_tables()
    xp = _array_namespace(t, p_mb, qv)
    dtype = t.dtype
    if dtype.kind != "f":
        raise TypeError("RRTM longwave inputs must be floating point")
    ncol, nz = t.shape
    profile = mm5atm_columns(
        xp, tables, t=t, p_mb=p_mb, dz=dz, qv=qv, qc=qc, qr=qr, qi=qi,
        qs=qs, qg=qg, cldfra=cldfra, tw=tw, pw_mb=pw_mb, tsfc=tsfc,
        emiss=emiss, r=r, g=g, co2vmr=co2vmr, n2ovmr=n2ovmr,
        ch4vmr=ch4vmr, nlayers=nlayers)
    coef = setcoef_columns(
        xp, tables, pavel=profile["pavel"], tavel=profile["tavel"],
        coldry=profile["coldry"], wkl=profile["wkl"])
    taug, pfrac = rrtm_taumol(
        coldry=profile["coldry"], wx=profile["wx"], tables=tables, **coef)

    #  GASABS's Pade lookup index, F:6746-6754.
    real = dtype.type
    odepth = RRTM_SECANG.astype(dtype) * taug
    tff = odepth / (RRTM_BPADE.astype(dtype) + odepth)
    tff = xp.where(odepth <= real(0.0), real(0.0), tff)
    itr = xp.clip(xp.trunc(real(5.0e3) * tff + real(0.5)).astype(xp.int32),
                  0, 5000)

    totdflux, totuflux, htr, htrc = rtrn_columns(
        xp, tables, tavel=profile["tavel"], pz=profile["pz"],
        tz=profile["tz"], cldfrac=profile["cldfrac"],
        taucloud=profile["taucloud"], itr=itr, pfrac=pfrac,
        tbound=profile["tbound"], semiss=profile["semiss"])

    #  F:2288-2297.  TTEN(K)=HTR(kte-K)/86400 in the top-down 1-D frame is
    #  HTR(k-1) for bottom-up level k, so the model-column heating is just
    #  the first nz entries of HTR.  GLW is the surface downward flux and
    #  OLR is TOTUFLUX at the model top -- not at NLAYERS.
    day = real(86400.0)
    return {
        "tten": htr[:, :nz] / day,
        "ttenc": htrc[:, :nz] / day,
        "glw": totdflux[:, 0],
        "olr": totuflux[:, nz],
        # Whole-column diagnostics, buffer layers included.  HTR is
        # K/day exactly as RTRN leaves it, so a caller can check the
        # flux-divergence identity the scheme is built on.
        "htr": htr,
        "totdflux": totdflux,
        "totuflux": totuflux,
        "pz": profile["pz"],
        # ``stages=True`` (the WRF column oracle's stage-by-stage
        # comparison) also hands back every intermediate RRTM's own
        # subroutines leave behind: MM5ATM's profile, SETCOEF's indices
        # and factors, GASABS's TAUG/PFRAC/ITR and RTRN's clear-sky HTRC.
        **({"stages": {**profile, **coef, "taug": taug, "pfrac": pfrac,
                       "itr": itr, "htrc": htrc}} if stages else {}),
    }


def _t8w_columns(xp, t3d, z_at_w, fnm, fnp):
    """Device twin of ``rrtmg_legacy._t8w_columns`` (phy_prep's ``t8w``).

    Same transcription, same source seam; this one stays on whatever
    array namespace the caller is using instead of forcing a host copy.
    """
    dtype = t3d.dtype
    real = dtype.type
    ncol, nz = t3d.shape
    z = real(0.5) * (z_at_w[:, :-1] + z_at_w[:, 1:])
    t8w = xp.empty((ncol, nz + 1), dtype=dtype)
    t8w[:, 1:nz] = (fnm[None, 1:nz] * t3d[:, 1:nz]
                    + fnp[None, 1:nz] * t3d[:, 0:nz - 1])
    w1 = (z_at_w[:, 0] - z[:, 1]) / (z[:, 0] - z[:, 1])
    t8w[:, 0] = w1 * t3d[:, 0] + (real(1.0) - w1) * t3d[:, 1]
    w1 = ((z_at_w[:, nz] - z[:, nz - 2]) / (z[:, nz - 1] - z[:, nz - 2]))
    t8w[:, nz] = w1 * t3d[:, nz - 1] + (real(1.0) - w1) * t3d[:, nz - 2]
    return t8w


def rrtmlwrad_columns(*, t3d, p3d, pi3d, dz8w, qv, qc, qr, qi, qs, qg,
                      cldfra, p8w, t8w, tsk, emiss, nlayers, co2vmr,
                      n2ovmr, ch4vmr, icloud=1, r=None, g=None, chunk=None,
                      tables=None, stages=False):
    """``RRTMLWRAD`` (F:1731-2020) for a block of WRF columns.

    The scheme's public boundary, in WRF's own layout: every input is
    BOTTOM-UP like the 3-D arrays the radiation driver hands RRTMLWRAD,
    ``(ncol, nz)`` for mass fields, ``(ncol, nz+1)`` for ``p8w``/``t8w``
    (Pa, K) and ``(ncol,)`` for ``tsk``/``emiss``.  ``cldfra`` is the
    driver's CLDFRA.  Returns ``rthraten``/``rthratenc`` (``(ncol, nz)``,
    bottom-up, K s-1 of potential temperature), ``glw`` and ``olr``.

    This is the seam the WRF column oracle grades
    (``tools/rrtm_lw_wrf461_oracle``) and the one the adapter below calls,
    so the packing it reproduces -- the top-down reversal, the hPa
    conversion, the hydrometeor clipping and the ``ICLOUD`` switch -- is
    measured, not assumed.  RRTMLWRAD's F_QI-absent branch (F:1929-1943)
    is not reproduced; see :meth:`RRTMLongwaveRadiation.
    longwave_with_cloud_fraction` for why it is unreachable here.
    """
    from woof.core import constants as c

    r = c.RD if r is None else r
    g = c.G if g is None else g
    xp = _array_namespace(t3d, p3d, qv)
    real = t3d.dtype.type
    zero = real(0.0)
    ncol, nz = t3d.shape

    def top_down(array):
        return xp.ascontiguousarray(array[:, ::-1])

    qv_td = top_down(xp.maximum(qv, zero))
    if icloud:
        cldfra_td = top_down(cldfra)
        qc_td = top_down(xp.maximum(qc, zero))
        qr_td = top_down(xp.maximum(qr, zero))
        qi_td = top_down(xp.maximum(qi, zero))
        qs_td = top_down(xp.maximum(qs, zero))
        qg_td = top_down(xp.maximum(qg, zero))
    else:
        #  RRTMLWRAD's "IF (ICLOUD .ne. 0)" block never runs, leaving
        #  every hydrometeor and the cloud fraction at zero (F:1888-1971).
        cldfra_td = xp.zeros((ncol, nz), dtype=t3d.dtype)
        qc_td = qr_td = qi_td = qs_td = qg_td = cldfra_td
    t_td = top_down(t3d)
    dz_td = top_down(dz8w)
    tw_td = top_down(t8w)
    # P1D = P3D/100. and Pw1D = p8w/100. (F:1886, 1909): a division.  The
    # adapter used to multiply by 0.01, which is not the same float32
    # operation; the oracle measured it one ULP off PAVEL and PZ.
    hundred = real(100.0)
    p_mb = top_down(p3d) / hundred
    pw_mb = top_down(p8w) / hundred

    tten = xp.empty((ncol, nz), dtype=t3d.dtype)
    ttenc = xp.empty((ncol, nz), dtype=t3d.dtype)
    glw = xp.empty((ncol,), dtype=t3d.dtype)
    olr = xp.empty((ncol,), dtype=t3d.dtype)
    step = ncol if chunk is None else max(int(chunk), 1)
    if stages and step < ncol:
        raise ValueError("stages=True needs the block in one chunk")
    for start in range(0, ncol, step):
        block = slice(start, min(start + step, ncol))
        result = rrtm_longwave_columns(
            t=t_td[block], p_mb=p_mb[block], dz=dz_td[block],
            qv=qv_td[block], qc=qc_td[block], qr=qr_td[block],
            qi=qi_td[block], qs=qs_td[block], qg=qg_td[block],
            cldfra=cldfra_td[block], tw=tw_td[block], pw_mb=pw_mb[block],
            tsfc=tsk[block], emiss=emiss[block], r=r, g=g,
            co2vmr=co2vmr, n2ovmr=n2ovmr, ch4vmr=ch4vmr, nlayers=nlayers,
            tables=tables, stages=stages)
        tten[block] = result["tten"]
        ttenc[block] = result["ttenc"]
        glw[block] = result["glw"]
        olr[block] = result["olr"]

    #  RRTMLWRAD F:2010-2013: rthraten += tten/pi3d, bottom-up.
    out = {"rthraten": tten / pi3d, "rthratenc": ttenc / pi3d,
           "glw": glw, "olr": olr}
    if stages:
        out["stages"] = {**result["stages"], "tten": tten, "ttenc": ttenc,
                         "totdflux": result["totdflux"],
                         "totuflux": result["totuflux"],
                         "htr": result["htr"]}
    return out


@dataclass
class RRTMLongwaveRadiation:
    """GPU-resident adapter for WRF ``ra_lw_physics=1``.

    This adapter owns the longwave half only: it publishes RTHRATENLW,
    GLW and OLR and returns zero shortwave.  It is therefore composable
    with any shortwave adapter -- see :class:`RRTMDudhiaRadiation` for the
    1/1 pairing WRF's ``lwrad_select CASE (1)`` / ``swrad_select CASE (1)``
    describes.

    GLW provenance is ``"scheme"``: RRTM computes the surface downward
    longwave from its own transfer and writes it, never reading whatever
    the field carried before.
    """

    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    p_top: float | None = None
    icloud: int = 1
    #: ``None`` auto-sizes the solver chunk from free device memory at
    #: the first EAGER solve of each ``(ncol, nlayers)`` geometry and
    #: caches it (:meth:`_column_chunk_for`); an integer pins it.  The
    #: answer is byte identical either way.
    column_chunk: int | None = None
    trace_gas_overrides: dict[str, float] | None = None
    #: WRF ghg_input: 1 reads CO2/N2O/CH4 from the CAMtr table at every
    #: call (RRTMLWRAD, module_ra_rrtm.F:1853-1854; REAL(8) values, used
    #: as such), 0 the fixed-gas formula.
    ghg_input: int = 0
    update_count: int = field(default=0, init=False)
    #: ``(ncol, nlayers) -> resolved chunk``, filled by eager solves only.
    #: Plain ints, so it never enters the restart classifier's array walk.
    _auto_chunk: dict = field(default_factory=dict, init=False, repr=False)

    publishes_olr = True
    #: The driver CLDFRA RRTMLWRAD radiated through (zeros under icloud=0).
    publishes_cldfra = True
    #: The GLW buffer is produced by this scheme, not carried in.
    glw_provenance = "scheme"

    def __post_init__(self) -> None:
        import cupy as cp

        if not isinstance(self.start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        # Coerce to a Python float exactly as the legacy-RRTMG sibling does
        # (woof/core/rrtmg_legacy.py): initialize_physics hands over
        # ``state.p_top``, a NumPy float32 SCALAR, which the restart
        # classifier's ``_is_array_like`` treats as an array -- an
        # unclassified one, so every RRTM run's end-of-run canonical
        # digest (and any restart write) died on a perfect integration
        # (1.9.1 D3).
        self.p_top = None if self.p_top is None else float(self.p_top)
        from woof.core.trace_gases import CLASSIC_GASES, validate_trace_gas_overrides
        if self.trace_gas_overrides is not None:
            self.trace_gas_overrides = validate_trace_gas_overrides(
                self.trace_gas_overrides, supported=CLASSIC_GASES,
                consumer="classic RRTM longwave")
        self.latitude_deg = cp.ascontiguousarray(
            cp.asarray(self.latitude_deg, dtype=cp.float32))
        self.longitude_deg = cp.ascontiguousarray(
            cp.asarray(self.longitude_deg, dtype=cp.float32))
        if self.latitude_deg.shape != self.longitude_deg.shape:
            raise ValueError("radiation latitude/longitude shapes must match")
        if self.icloud not in (0, 1, 2):
            raise ValueError("icloud=3 requires unported cal_cldfra3 condensate adjustment; choose 0, 1 or 2")
        if self.column_chunk is not None:
            effective_column_chunk(self.column_chunk, 1)  # validate the explicit pin
        # Fail closed at construction if the packaged tables are missing
        # or corrupt, rather than mid-forecast.
        self._tables = load_rrtm_lw_tables()

    @staticmethod
    def _columns(array):
        """``(nz, ny, nx)`` model field -> ``(ncol, nz)`` BOTTOM-UP columns."""
        import cupy as cp
        nz, ny, nx = array.shape
        return cp.ascontiguousarray(
            array.transpose(1, 2, 0).reshape(ny * nx, nz))

    @staticmethod
    def _state_field(state, name, fallback):
        value = getattr(state, name, None)
        return fallback if value is None else value

    def _resolve_p_top(self, atmosphere, state) -> float:
        if self.p_top is not None:
            return float(self.p_top)
        declared = getattr(state, "p_top", None)
        if declared is not None:
            return float(declared)
        return float(atmosphere["p_interface"][-1].max())

    def _column_chunk_for(self, ncol: int, nlayers: int, cp) -> int:
        """The solver chunk: the pin, cached auto-size, or capture fallback.

        The auto-size reads ``cudaMemGetInfo``, which is ILLEGAL inside
        CUDA graph capture -- the 8d94a2ff7 perf fix called it on every
        solve and took both graph-replay rungs of the tiles gate down
        with ``GraphCaptureError``.  So the memory query runs on EAGER
        solves only, once per ``(ncol, nlayers)`` geometry, and its
        answer is cached; a captured solve reuses the cached value.  A
        capture that arrives COLD (no eager solve ran first) uses the
        ``DEFAULT_COLUMN_CHUNK`` floor without caching it, so a later
        eager call still auto-sizes.  Every branch returns the same
        bytes: chunking is per-column arithmetic only
        (tests/test_rrtm_longwave.py::
        test_column_chunking_does_not_change_the_answer), so this choice
        is throughput, never trajectory.  An explicit ``column_chunk``
        pins all of it, exactly as before.
        """
        if self.column_chunk is not None:
            return int(self.column_chunk)
        key = (int(ncol), int(nlayers))
        cached = self._auto_chunk.get(key)
        if cached is not None:
            return cached
        if _stream_is_capturing(cp):
            return min(int(ncol), DEFAULT_COLUMN_CHUNK)
        resolved = _resolve_column_chunk(None, ncol, nlayers, cp)
        self._auto_chunk[key] = resolved
        return resolved

    def longwave(self, *, atmosphere, fields, state, cfg):
        """Run RRTM LW and return ``(rthratenlw, glw, olr)`` on device."""
        return self.longwave_with_cloud_fraction(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)[:3]

    def longwave_with_cloud_fraction(self, *, atmosphere, fields, state, cfg):
        """:meth:`longwave` plus the driver CLDFRA it radiated through.

        Returns ``(rthratenlw, glw, olr, cldfra)``; ``cldfra`` is the
        (nz, ny, nx) bottom-up icloud=1 fraction after the MYNN merge
        (zeros under icloud=0, where RRTMLWRAD's cloud block never runs).
        """
        import cupy as cp
        from woof.core import constants as c
        from woof.core.mynn_radiation import (merge_mynn_bl_clouds,
                                               mynn_bl_cloud_active,
                                               wrf_itimestep)
        from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
        from woof.core.rrtmgp import cal_cldfra1, cal_cldfra2

        temperature = atmosphere["temperature"]
        nz, ny, nx = temperature.shape
        if self.latitude_deg.shape != (ny, nx):
            raise ValueError(
                "radiation latitude/longitude must match state grid")
        ncol = ny * nx
        nlayers = rrtm_layer_count(nz, self._resolve_p_top(atmosphere, state))

        zero = cp.zeros_like(temperature)
        #  Bottom-up (ncol, nz) columns: WRF's own 3-D layout, which is what
        #  rrtmlwrad_columns takes.  It owns the top-down reversal.
        qv = self._columns(cp.maximum(atmosphere["qv"], cp.float32(0.0)))
        qc = self._columns(cp.maximum(
            self._state_field(state, "qc", atmosphere["qc"]),
            cp.float32(0.0)))
        qr = self._columns(cp.maximum(
            self._state_field(state, "qr", zero), cp.float32(0.0)))
        qi = self._columns(cp.maximum(
            self._state_field(state, "qi", atmosphere["qi"]),
            cp.float32(0.0)))
        qs = self._columns(cp.maximum(
            self._state_field(state, "qs", zero), cp.float32(0.0)))
        qg = self._columns(cp.maximum(
            self._state_field(state, "qg", zero), cp.float32(0.0)))
        t_cols = self._columns(temperature)
        p_cols = self._columns(atmosphere["pressure"])
        dz_cols = self._columns(atmosphere["dz"])

        if self.icloud:
            # cal_cldfra1 is the driver's icloud=1 branch and takes
            # (ncol, nlay) blocks; it is elementwise, so the column
            # orientation does not enter it.
            #
            # F_QI/F_QS come from the scheme's Registry package membership
            # (the same resolution the legacy-RRTMG sibling performs),
            # never hardcoded True.  WRF's driver hands cal_cldfra1 the
            # Registry flags (module_radiation_driver.F:1326-1332) and the
            # qc-only arm at :3891-3899 weights a sub-freezing liquid
            # cloud with the ICE saturation (weight=1 at T<=273.15 K).
            # Hardcoding f_qi=f_qs=True with zero-filled qi/qs sent a
            # Kessler-class supercooled cloud through the Morrison arm at
            # :3870-3877 instead, whose weight collapses to 0 (liquid
            # saturation) -- a real divergence on this default path.  For
            # a both-species scheme the zero-fill made the two spellings
            # identical, and for P3 (qi, no qs) the qs=0 collapse makes
            # (True, True) equal (True, False) byte for byte
            # (tests/test_rrtm_lw_cloud_fraction_flags.py measures it).
            f_qi, f_qs = legacy_cloud_fraction_flags(
                int(getattr(cfg, "mp_physics", 0)))
            if self.icloud == 2:
                cldfra = cal_cldfra2(qc, qi,
                    f_qc=int(getattr(cfg, "mp_physics", 0)) != 0, f_qi=f_qi)
            else:
                cldfra = cal_cldfra1(qv, qc, qi, qs, t_cols, p_cols,
                                     f_qc=True, f_qi=f_qi, f_qs=f_qs,
                                     mp_physics=int(getattr(cfg, "mp_physics", 0)))
            active_bl = self.icloud == 1 and mynn_bl_cloud_active(
                getattr(cfg, "bl_pbl_physics", 0),
                getattr(cfg, "icloud_bl", 0))
            qc, qi, cldfra = merge_mynn_bl_clouds(
                qc, qi, cldfra,
                qc_bl=self._columns(fields["qc_bl"]) if active_bl else None,
                qi_bl=self._columns(fields["qi_bl"]) if active_bl else None,
                cldfra_bl=(self._columns(fields["cldfra_bl"])
                           if active_bl else None),
                bl_pbl_physics=getattr(cfg, "bl_pbl_physics", 0),
                icloud_bl=(getattr(cfg, "icloud_bl", 0) if self.icloud == 1 else 0),
                itimestep=(wrf_itimestep(state.elapsed_seconds, cfg.dt)
                           if active_bl else 1),
                bl_mynn_version=getattr(cfg, "bl_mynn_version", "wrf_461"),
                temperature=t_cols if active_bl else None)
            #  RRTMLWRAD's F_QI-absent branch (F:1929-1943), which moves
            #  QC into QI and QR into QS below 273.15 K, is deliberately
            #  not reproduced: it exists for schemes that do not carry
            #  ice, and woof's state always presents QI/QS -- zero-filled
            #  for a warm-rain scheme, which is the same result WRF
            #  reaches through its warm_rain guard.
        else:
            #  icloud=0: rrtmlwrad_columns zeroes every hydrometeor and the
            #  fraction itself (RRTMLWRAD F:1888-1971); the published
            #  CLDFRA is zero too.
            cldfra = cp.zeros_like(t_cols)

        p8w = self._columns(atmosphere["p_interface"])
        z_at_w = self._columns(atmosphere["z_interface"])
        t8w = _t8w_columns(
            cp, t_cols, z_at_w,
            cp.asarray(state.fnm, dtype=cp.float32),
            cp.asarray(state.fnp, dtype=cp.float32))

        tsk = cp.asarray(fields["tsk"], dtype=cp.float32).reshape(-1)
        emiss = cp.asarray(fields["emiss"], dtype=cp.float32).reshape(-1)
        pi3d = self._columns(atmosphere["exner"])

        valid_time = (self.start_time
                      + timedelta(seconds=float(state.elapsed_seconds)))
        gases = rrtm_default_trace_gases(valid_time.year, self.trace_gas_overrides)
        if int(self.ghg_input) == 1:
            from woof.core.camtr_gases import camtr_trace_gases
            hour = (valid_time.hour + valid_time.minute / 60.0
                    + valid_time.second / 3600.0 + valid_time.microsecond / 3.6e9)
            julian = np.float32((valid_time.timetuple().tm_yday - 1) + hour / 24.0)
            camtr = camtr_trace_gases(valid_time.year, julian)
            gases = {**gases, **{name: camtr[name] for name in ("co2", "n2o", "ch4")},
                     **(self.trace_gas_overrides or {})}

        result = rrtmlwrad_columns(
            t3d=t_cols, p3d=p_cols, pi3d=pi3d, dz8w=dz_cols, qv=qv, qc=qc,
            qr=qr, qi=qi, qs=qs, qg=qg, cldfra=cldfra, p8w=p8w, t8w=t8w,
            tsk=tsk, emiss=emiss, nlayers=nlayers, co2vmr=gases["co2"],
            n2ovmr=gases["n2o"], ch4vmr=gases["ch4"], icloud=self.icloud,
            r=c.RD, g=c.G, chunk=self._column_chunk_for(ncol, nlayers, cp),
            tables=self._tables)

        rthratenlw = cp.ascontiguousarray(
            result["rthraten"].reshape(ny, nx, nz).transpose(2, 0, 1))
        self.update_count += 1
        cloud = cp.ascontiguousarray(
            cldfra.reshape(ny, nx, nz).transpose(2, 0, 1))
        glw, olr = result["glw"], result["olr"]
        return rthratenlw, glw.reshape(ny, nx), olr.reshape(ny, nx), cloud

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp
        from woof.core.physics import RadiationResult

        rthratenlw, glw, olr, cldfra = self.longwave_with_cloud_fraction(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        return RadiationResult(
            rthratenlw=rthratenlw,
            rthratensw=cp.zeros_like(rthratenlw),
            swdown=cp.zeros(glw.shape, dtype=cp.float32),
            glw=glw,
            gsw=cp.zeros(glw.shape, dtype=cp.float32),
            coszen=cp.zeros(glw.shape, dtype=cp.float32),
            olr=olr, cldfra=cldfra)

    @property
    def restart_identity(self) -> dict[str, object]:
        """Trajectory-changing setup bound into tree restart identity."""
        identity = {
            "algorithm": "wrf-v4.6.1-rrtm-longwave",
            "wrf_source": "phys/module_ra_rrtm.F:RRTMLWRAD/RRTM",
            "icloud": int(self.icloud),
            "ghg_input": int(self.ghg_input),
            "supported_path": ("no-chem,no-cam-gases,o3-climatology"
                               if not int(self.ghg_input) else
                               "no-chem,camtr-gases,o3-climatology"),
        }
        if self.trace_gas_overrides:
            identity["trace_gas_overrides"] = dict(self.trace_gas_overrides)
        return identity


@dataclass
class RRTMDudhiaRadiation:
    """WRF's classic ``ra_lw_physics=1`` + ``ra_sw_physics=1`` pairing.

    The two WRF schemes are independent: ``module_radiation_driver.F``
    dispatches ``lwrad_select`` and ``swrad_select`` separately and each
    accumulates into its own tendency.  woof's other radiation options
    are coupled adapters that own both streams, which is why no
    longwave-on/Dudhia-shortwave combination existed before.  This class
    is that composition and nothing more: it runs
    :class:`RRTMLongwaveRadiation` and
    :class:`~woof.core.dudhia.DudhiaShortwaveRadiation` and merges their
    results, leaving both single-stream adapters untouched.

    GLW comes from RRTM (provenance ``"scheme"``), so the composed result
    never depends on what the GLW field held on entry -- which is the
    property that makes the pairing safe whichever way the surrounding
    GLW-defaulting work lands.
    """

    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    p_top: float | None = None
    icloud: int = 1
    swrad_scat: float = 1.0
    #: Forwarded to :class:`RRTMLongwaveRadiation`: ``None`` auto-sizes.
    column_chunk: int | None = None
    trace_gas_overrides: dict[str, float] | None = None
    ghg_input: int = 0

    publishes_olr = True
    publishes_cldfra = True
    glw_provenance = "scheme"

    def __post_init__(self) -> None:
        from woof.core.dudhia import DudhiaShortwaveRadiation

        # Same coercion as the longwave adapter below performs for its
        # own copy: this dataclass retains ``p_top`` as a direct
        # attribute, and a NumPy float32 scalar here is an unclassified
        # "array" to the restart classifier (1.9.1 D3).
        self.p_top = None if self.p_top is None else float(self.p_top)
        from woof.core.trace_gases import CLASSIC_GASES, validate_trace_gas_overrides
        if self.trace_gas_overrides is not None:
            self.trace_gas_overrides = validate_trace_gas_overrides(
                self.trace_gas_overrides, supported=CLASSIC_GASES,
                consumer="classic RRTM longwave")
        self.longwave_adapter = RRTMLongwaveRadiation(
            self.start_time, self.latitude_deg, self.longitude_deg,
            p_top=self.p_top, icloud=self.icloud,
            column_chunk=self.column_chunk, trace_gas_overrides=self.trace_gas_overrides,
            ghg_input=self.ghg_input)
        self.shortwave_adapter = DudhiaShortwaveRadiation(
            self.start_time, self.latitude_deg, self.longitude_deg,
            swrad_scat=self.swrad_scat, icloud=self.icloud)

    @property
    def spectrum_adapters(self):
        return (self.longwave_adapter, self.shortwave_adapter)

    @property
    def update_count(self) -> int:
        return self.longwave_adapter.update_count

    def __call__(self, *, atmosphere, fields, state, cfg):
        from woof.core.physics import RadiationResult

        rthratenlw, glw, olr, cldfra = (
            self.longwave_adapter.longwave_with_cloud_fraction(
                atmosphere=atmosphere, fields=fields, state=state, cfg=cfg))
        shortwave = self.shortwave_adapter(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
        return RadiationResult(
            rthratenlw=rthratenlw,
            rthratensw=shortwave.rthratensw,
            swdown=shortwave.swdown,
            glw=glw,
            gsw=shortwave.gsw,
            coszen=shortwave.coszen,
            olr=olr, cldfra=cldfra)

    @property
    def restart_identity(self) -> dict[str, object]:
        return {
            "algorithm": "wrf-v4.6.1-rrtm-lw+dudhia-sw",
            "longwave": self.longwave_adapter.restart_identity,
            "shortwave": self.shortwave_adapter.restart_identity,
        }


__all__ = ["DEFAULT_COLUMN_CHUNK", "RRTMDudhiaRadiation",
           "RRTMLongwaveRadiation", "mm5atm_columns", "o3data_columns",
           "rrtm_layer_count", "rrtm_longwave_columns", "rrtmlwrad_columns",
           "rtrn_columns", "setcoef_columns"]
