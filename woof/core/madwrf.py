"""MAD-WRF: passive-hydrometeor microphysics (mp_physics = 96), its cloud
initialization and its hydrometeor-tracer nudging.

Port of WRF v4.7.1 (f52c197ed39d12e087d02c50f412d90d418f6186).
The cloud module is byte-identical to WRF v4.6.1; the earlier port is reused.
WRF ships this code under its public-domain statement.

* ``phys/module_madwrf.F`` (994 lines).  ``Init_madwrf_clouds`` (:71-204)
  is real.exe's ``madwrf_cldinit = 1`` (module_initialize_real.F:4861-4865):
  per column it converts theta to temperature, derives layer depths and a
  tropopause, optionally imposes an observed cloud mask / cloud-top height
  (from ``cldtopz`` or from a satellite brightness temperature) / cloud-base
  height, and calls ``cal_cldfra3_madwrf`` (:241-558), WRF's
  RH-based cloud fraction with ``find_cloudLayers`` (:633-797) inserting
  liquid and ice where the fraction says cloud and the analysis has none,
  ``adjust_cloudFinal`` (:955-992) capping the inserted water path, and the
  vapour raised to saturation in partly cloudy layers.
  ``Init_madwrf_tracers`` (:206-237) copies qc/qi/qs into the
  ``tr_qc/tr_qi/tr_qs`` tracers (``madwrf_opt = 2``, module_initialize_real
  .F:4868-4873).
* ``dyn_em/module_big_step_utilities_em.F:6711-6756`` ``Cloud_tracer_nudge``,
  called from solve_em.F:3641-3656 before moist_physics_prep while
  ``xtime < madwrf_dt_nudge`` minutes: ``q += (tr - q) * dt / dt_relax``.
* ``mp_physics = 96`` itself: the microphysics driver's ``CASE (MADWRF_MP)``
  (module_microphysics_driver.F:2895-2896) is EMPTY.  The scheme is the
  Registry package ``madwrf_mp mp_physics==96 - moist:qv,qc,qi,qs``
  (Registry.EM_COMMON:3052): hydrometeors that are only advected and mixed.

Two implementations, one arithmetic.  The NumPy functions here are the
host path of the real initialization and the mirror the WRF column oracle
grades (tools/madwrf_wrf471_oracle); ``kernels/madwrf.cu`` is the device
path.  Both evaluate WRF's FP32 operation order, and both take the one libm
word the module uses -- ``REAL ** REAL`` -- from WOOF's own float32 powf
(``woof.core.noahmp_libm.powf_array`` on the host, ``gfk_pow`` in
``glibc_flt32.cuh`` on the card).  SQRT is correctly rounded on both sides.
The device kernels compile on the no-FTZ, no-FMA route
(``--ftz=false --fmad=false``), so subnormal hydrometeor words survive the
nudge exactly as they do in WRF.

DECLARED DIVERGENCE (one).  ``find_k_from_z_agl`` (:560-584) loops
``k = kts, kte`` and reads ``dz(kte)``; Init_madwrf_clouds passes ``dzs``,
an automatic array dimensioned ``kts:kte-1`` (:97), so that read is one
element past the array and its value is whatever the stack holds.  It
matters only for an observed cloud top or base ABOVE the model top: WRF
then returns a level index that depends on stack contents (``kte`` or
the caller's -999), and the ``cldtopk < cldbasek`` test at :362 decides
differently for the two.  The port searches the model's own layers only,
so an observation above the model top finds no level (-999), which is
what every in-bounds read path returns.  The tested in-column observations are
bit-identical to the pinned WRF oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from woof.core.device_cache import cuda_cache

F32 = np.float32

#: share/module_model_constants.F:17/:19-20/:31/:37.
G = F32(9.81)
R_D = F32(287.0)
CP = F32(7.0) * R_D / F32(2.0)
RCP = R_D / CP
T0 = F32(300.0)
#: module_madwrf.F:666/:828 exponent, folded in FP32 exactly as gfortran
#: folds the default-REAL constant expression ``287.05/1004.``.
THETA_EXPONENT = F32(287.05) / F32(1004.0)
#: :837/:846 ``10./1500.``
TROPO_LAPSE = F32(10.0) / F32(1500.0)
MISSING = F32(-999.9)            # :48, :96 MISSING_* parameters
CLDFRA_DEF = F32(0.5)            # :272
CLDFRA_THRESH = F32(0.0)         # :273
RH_NOCLOUD = F32(0.3)            # :274
MAX_RELH = F32(1.5)              # Init_madwrf_clouds passes 1.5 (:162)
ENTRMNT = F32(0.5)               # :262
NO_LEVEL = -999                  # :300-303


def _powf(base, exponent):
    from woof.core.noahmp_libm import powf_array
    return powf_array(np.asarray(base, dtype=F32), F32(exponent))


def rslf(p, t):
    """module_mp_thompson.F:5383-5413 RSLF, FP32 Horner order."""
    c0, c1, c2, c3 = F32(.611583699E03), F32(.444606896E02), \
        F32(.143177157E01), F32(.264224321E-1)
    c4, c5, c6, c7 = F32(.299291081E-3), F32(.203154182E-5), \
        F32(.702620698E-8), F32(.379534310E-11)
    c8 = F32(-.321582393E-13)
    x = np.maximum(F32(-80.0), t - F32(273.16))
    esl = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (
        c5 + x * (c6 + x * (c7 + x * c8)))))))
    esl = np.minimum(esl, p * F32(0.15))
    return F32(.622) * esl / (p - esl)


def rsif(p, t):
    """module_mp_thompson.F:5418-5444 RSIF, FP32 Horner order."""
    c0, c1, c2, c3 = F32(.609868993E03), F32(.499320233E02), \
        F32(.184672631E01), F32(.402737184E-1)
    c4, c5, c6, c7 = F32(.565392987E-3), F32(.521693933E-5), \
        F32(.307839583E-7), F32(.105785160E-9)
    c8 = F32(.161444444E-12)
    x = np.maximum(F32(-80.0), t - F32(273.16))
    esi = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (
        c5 + x * (c6 + x * (c7 + x * c8)))))))
    esi = np.minimum(esi, p * F32(0.15))
    return F32(.622) * esi / np.maximum(F32(1.e-4), p - esi)


# --------------------------------------------------------------------------
# Column helpers.  Arrays are (level, column) with level index k-1 for
# WRF's 1-based k; integer level variables keep WRF's 1-based values.
# ``nl`` is the number of mass layers, so WRF's kte is nl + 1.
# --------------------------------------------------------------------------

def _at(arr, k):
    """arr(k) per column for a per-column 1-based level vector ``k``."""
    cols = np.arange(arr.shape[1])
    return arr[np.clip(k - 1, 0, arr.shape[0] - 1), cols]


def _seq_sum(arr, k_lo, k_hi):
    """Fortran SUM(arr(k_lo:k_hi)) per column, accumulated in order."""
    out = np.zeros(arr.shape[1], dtype=F32)
    for k in range(1, arr.shape[0] + 1):
        sel = (k >= k_lo) & (k <= k_hi)
        out = np.where(sel, out + arr[k - 1], out)
    return out


def _theta(t, p):
    return t * _powf(F32(100000.0) / p, THETA_EXPONENT)


def calc_tropo_height(t, p, dz, theta=None):
    """module_madwrf.F:799-868 Calc_tropo_height -> (k_tropo, tropo_z)."""
    nl, ncol = t.shape
    kte = nl + 1
    if theta is None:
        theta = _theta(t, p)
    k_p200 = np.zeros(ncol, dtype=np.int64)
    for k in range(kte - 1, 0, -1):
        k_p200 = np.where((p[k - 1] > F32(19999.0)) & (k_p200 == 0),
                          k, k_p200)
    k_p200 = np.where((kte - k_p200) < 3, kte - 3, k_p200)

    def delz_at(k):
        return (F32(0.5) * _at(dz, k) + _at(dz, k + 1)) \
            + F32(0.5) * _at(dz, k + 2)

    # DO k = k_p200-2, kts, -1 ... EXIT: a completed loop leaves k = kts-1,
    # a zero-trip loop leaves k at its start value.
    start = k_p200 - 2
    kk = np.where(start >= 1, 0, start)
    open_ = start >= 1
    for k in range(int(start.max()), 0, -1):
        live = open_ & (k <= start)
        if not live.any():
            continue
        theta1 = _at(theta, np.full(ncol, k))
        theta2 = _at(theta, np.full(ncol, k + 2))
        delz = delz_at(np.full(ncol, k))
        stop = live & ((((theta2 - theta1) / delz) < TROPO_LAPSE)
                       | (p[k - 1] > F32(70000.0)))
        kk = np.where(stop, k, kk)
        open_ = open_ & ~stop
    k_tropo = np.maximum(3, np.minimum(kk + 2, kte - 1))
    second = k_tropo > k_p200
    if second.any():
        lo = k_p200 - 2
        first = kte - 3
        kk2 = np.where(first >= lo, lo - 1, first)
        open2 = second & (first >= lo)
        for k in range(first, 0, -1):
            live = open2 & (k >= lo)
            if not live.any():
                continue
            theta1 = theta[k - 1]
            theta2 = theta[k + 1]
            delz = delz_at(np.full(ncol, k))
            stop = live & ((((theta2 - theta1) / delz) < TROPO_LAPSE)
                           & (p[k - 1] > F32(9500.0)))
            kk2 = np.where(stop, k, kk2)
            open2 = open2 & ~stop
        k_tropo = np.where(second,
                           np.maximum(k_p200 - 1,
                                      np.minimum(kk2 + 2, kte - 1)),
                           k_tropo)
    tropo_z = _seq_sum(dz, np.ones(ncol, dtype=np.int64), k_tropo)
    return k_tropo, tropo_z


def calc_cldtopz_from_brtemp(cldmask, ts, dzs, brtemp, tropoz, ht):
    """module_madwrf.F:23-69 Calc_cldtopz_from_brtemp."""
    nl, ncol = ts.shape
    kte = nl + 1
    out = np.full(ncol, MISSING, dtype=F32)
    live = (brtemp != MISSING) & (cldmask > F32(0.0))
    height = _seq_sum(dzs, np.ones(ncol, dtype=np.int64),
                      np.full(ncol, kte - 1)) + ht
    for k in range(kte - 2, 0, -1):
        if not live.any():
            break
        height = np.where(live, height - dzs[k], height)
        test = live & ~(height > tropoz) & (ts[k - 1] > brtemp)
        value = height + dzs[k - 1] * (brtemp - ts[k - 1]) \
            / (ts[k] - ts[k - 1])
        value = np.where((value < height) | (value > height + dzs[k - 1]),
                         tropoz, value)
        out = np.where(test, value, out)
        live = live & ~test
    return out


def find_k_from_z_agl(z_agl, dz, k_lev):
    """module_madwrf.F:560-584, over the model's own layers (declared
    divergence: WRF's k = kte iteration reads past ``dzs``)."""
    k_lev = k_lev.copy()
    z_full = np.zeros(dz.shape[1], dtype=F32)
    for k in range(1, dz.shape[0] + 1):
        z_this = z_full
        z_next = z_this + dz[k - 1]
        hit = (z_agl > F32(0.0)) & (z_agl > z_this) & (z_agl <= z_next)
        k_lev = np.where(hit, k, k_lev)
        z_full = z_next
    return k_lev


@dataclass
class ColumnCounts:
    """Branch census of one call, for coverage claims in tests."""
    masked_clear: int = 0
    cldtopk_found: int = 0
    cldbasek_found: int = 0
    ocean_rh_nocloud: int = 0
    rh_cloud_fraction: int = 0
    cloud_extended_to_top: int = 0
    cloud_added_above_base: int = 0
    ice_layer: int = 0
    ice_single: int = 0
    water_layer: int = 0
    water_single: int = 0
    lwp_capped: int = 0
    iwp_capped: int = 0
    qv_raised: int = 0
    stratospheric_removed: int = 0
    below_lcl_halved: int = 0


def _adjust_cloud(cfr, q, qs, qvs, t, dz, k1, k2, active, t_min, counts_key,
                  counts):
    """adjust_cloudIce (:870-907) / adjust_cloudH2O (:911-948).

    ``qs`` is None for the water form (its max_lwc subtracts qc only)."""
    nl = cfr.shape[0]
    tdz = np.zeros(cfr.shape[1], dtype=F32)
    for k in range(1, nl + 1):
        sel = active & (k >= k1) & (k <= k2)
        tdz = np.where(sel, tdz + dz[k - 1], tdz)
    max_c = np.abs(_at(qvs, k2) - _at(qvs, k1))
    for k in range(1, nl + 1):
        sel = active & (k >= k1) & (k <= k2)
        sub = q[k - 1] + qs[k - 1] if qs is not None else q[k - 1]
        max_c = np.where(sel, np.maximum(F32(1.E-6), max_c - sub), max_c)
    max_c = np.minimum(F32(1.E-3), max_c)
    this_dz = np.zeros(cfr.shape[1], dtype=F32)
    for k in range(1, nl + 1):
        sel = active & (k >= k1) & (k <= k2)
        if not sel.any():
            continue
        this_dz = np.where(sel, np.where(k == k1, this_dz + F32(0.5)
                                         * dz[k - 1], this_dz + dz[k - 1]),
                           this_dz)
        this_c = max_c * this_dz / tdz
        add = np.maximum(F32(1.E-6), this_c * (F32(1.0) - ENTRMNT))
        c = cfr[k - 1]
        put = sel & (c > F32(0.0)) & (c < F32(1.0)) & (t[k - 1] >= t_min)
        q[k - 1] = np.where(put, q[k - 1] + c * c * add, q[k - 1])
    setattr(counts, counts_key,
            getattr(counts, counts_key) + int(active.sum()))


def _layer_scan(cfr, qvs, t, dz, q, qs, k_start, k_floor, k_cldb0, ice,
                counts):
    """The two DO WHILE scans of find_cloudLayers (:722-759, :762-795)."""
    ncol = cfr.shape[1]
    k = k_start.copy()
    k_cldb = k_cldb0.copy()
    floor_plus = k_floor + 1 if ice else k_floor
    # ice: DO WHILE (k > k_m12C+1), inner floor k_m12C
    # water: DO WHILE (k > kbot), inner floor kbot
    live = k > floor_plus
    while live.any():
        c = _at(cfr, k)
        cloudy = live & (c >= F32(0.01))
        k_cldt = np.where(cloudy, k, 0)
        new_b = k_cldb.copy()
        found = np.zeros(ncol, dtype=bool)
        if cloudy.any():
            top = int(k_cldt.max())
            for k2 in range(top - 1, 0, -1):
                sel = cloudy & ~found & (k2 <= k_cldt - 1) & (k2 >= k_floor)
                if not sel.any():
                    continue
                hit = sel & ((cfr[k2 - 1] < F32(0.01)) | (k2 == k_floor))
                new_b = np.where(hit, k2 + 1, new_b)
                found = found | hit
        k_cldb = np.where(cloudy, new_b, k_cldb)
        depth = k_cldt - k_cldb + 1
        layer = live & (depth >= 2)
        single = live & (depth == 1)
        if layer.any():
            _adjust_cloud(cfr, q, qs if ice else None, qvs, t, dz, k_cldb,
                          k_cldt, layer, F32(203.16) if ice else F32(253.16),
                          "ice_layer" if ice else "water_layer", counts)
        if single.any():
            cb = _at(cfr, k_cldb)
            put = single & (cb > F32(0.0)) & (cb < F32(1.0))
            qv_s = _at(qvs, k_cldb)
            value = F32(0.05) * qv_s * cb * cb
            idx = np.clip(k_cldb - 1, 0, cfr.shape[0] - 1)
            cols = np.arange(ncol)
            cur = q[idx, cols]
            q[idx, cols] = np.where(put, value, cur)
            key = "ice_single" if ice else "water_single"
            setattr(counts, key, getattr(counts, key) + int(put.sum()))
        k = np.where(layer | single, k_cldb, k)
        k = np.where(live, k - 1, k)
        live = live & (k > floor_plus)


def find_cloud_layers(qvs, cfr, t, p, dz, qc, qi, qs, keep_below_lcl,
                      counts):
    """module_madwrf.F:633-797 find_cloudLayers (no ktropo argument)."""
    nl, ncol = t.shape
    kte = nl + 1
    theta = _theta(t, p)
    k_m12 = np.zeros(ncol, dtype=np.int64)
    k_m40 = np.zeros(ncol, dtype=np.int64)
    for k in range(kte - 1, 0, -1):
        tc = t[k - 1] - F32(273.16)
        k_m40 = np.where((tc > F32(-40.0)) & (p[k - 1] > F32(7000.0)),
                         np.maximum(k_m40, k), k_m40)
        k_m12 = np.where((tc > F32(-12.0)) & (p[k - 1] > F32(10100.0)),
                         np.maximum(k_m12, k), k_m12)
    k_m40 = np.where(k_m40 <= 1, 1, k_m40)
    k_m12 = np.where(k_m12 <= 1, 1, k_m12)
    bad = (k_m40 > kte - 2) | (k_m12 > kte - 3)
    if bad.any():
        raise ValueError(
            "MAD-WRF cloud initialization: a column's -40 C or -12 C level "
            "lies within two/three layers of the model top, where WRF's "
            "find_cloudLayers stops the run (module_madwrf.F:673-681, "
            "'problem in temperature profile'): the layer scans above it "
            "would index past the column. Raise the model top or check the "
            "analysis temperature profile.")
    k_tropo, _ = calc_tropo_height(t, p, dz, theta=theta)
    for k in range(1, kte):
        sel = (k >= k_tropo + 1) & (cfr[k - 1] > F32(0.0)) \
            & (cfr[k - 1] < F32(1.0))
        counts.stratospheric_removed += int(sel.sum())
        cfr[k - 1] = np.where(sel, F32(0.0), cfr[k - 1])
    # DO k = kbot, k_m12C ... EXIT; completed -> k_m12C+1, zero-trip -> kbot.
    kbot0 = 3
    kk = np.where(k_m12 >= kbot0, k_m12 + 1, kbot0)
    open_ = k_m12 >= kbot0
    for k in range(kbot0, int(k_m12.max()) + 1):
        live = open_ & (k <= k_m12)
        if not live.any():
            continue
        stop = live & ((theta[k - 1] - theta[k - 2])
                       > F32(0.010E-3) * dz[k - 1])
        kk = np.where(stop, k, kk)
        open_ = open_ & ~stop
    kbot = np.maximum(2, kk - 2)
    halve = ~keep_below_lcl
    for k in range(1, kte):
        sel = halve & (k <= kbot) & (cfr[k - 1] > F32(0.0)) \
            & (cfr[k - 1] < F32(1.0))
        counts.below_lcl_halved += int(sel.sum())
        cfr[k - 1] = np.where(sel, F32(0.5) * cfr[k - 1], cfr[k - 1])
    kbot = np.where(keep_below_lcl, 2, kbot)
    _layer_scan(cfr, qvs, t, dz, qi, qs, k_tropo, k_m12, k_tropo, True,
                counts)
    _layer_scan(cfr, qvs, t, dz, qc, None, k_m12 + 2, kbot, k_m12 + 3,
                False, counts)


def cal_cldfra3_madwrf(cldfra, qv, qc, qi, qs, dz, p, t, xland, gridkm,
                       mode, cldmask, cldtopz, cldbasez, counts):
    """module_madwrf.F:241-558 with modify_qvapor=.true., max_relh=1.5 and
    no k_tropo, as Init_madwrf_clouds calls it.

    ``mode`` is Init_madwrf_clouds' insert_clouds (0-3) per column: 1 passes
    cldmask, 2 adds cldtopz, 3 adds cldbasez.  Arrays are (level, column);
    ``cldfra`` has one level more than the others (WRF's kte).
    """
    nl, ncol = t.shape
    kte = nl + 1
    imp_mask = mode == 1
    imp_top = mode == 2
    imp_base = mode == 3
    imp_any_top = imp_top | imp_base
    cldtopk = np.full(ncol, NO_LEVEL, dtype=np.int64)
    cldbasek = np.full(ncol, NO_LEVEL, dtype=np.int64)
    done = np.zeros(ncol, dtype=bool)

    def zero_levels(sel, lo, hi):
        for k in range(1, kte):
            s = sel & (k >= lo) & (k <= hi)
            for arr in (cldfra, qc, qi, qs):
                arr[k - 1] = np.where(s, F32(0.0), arr[k - 1])

    clear = (imp_mask | imp_any_top) & (cldmask == F32(0.0))
    counts.masked_clear += int(clear.sum())
    zero_levels(clear, 1, kte - 1)
    done |= clear
    obs_ok = (cldmask > F32(0.0)) & (cldmask <= F32(1.0))
    top_try = ~done & imp_any_top & obs_ok & (cldtopz > F32(0.0))
    cldtopk = np.where(top_try, find_k_from_z_agl(cldtopz, dz, cldtopk),
                       cldtopk)
    top_ok = top_try & (cldtopk > 0) & (cldtopk < kte)
    counts.cldtopk_found += int(top_ok.sum())
    zero_levels(top_ok, cldtopk + 1, kte - 1)
    base_try = ~done & imp_base & obs_ok & (cldbasez > F32(0.0))
    cldbasek = np.where(base_try, find_k_from_z_agl(cldbasez, dz, cldbasek),
                        cldbasek)
    cldbasek = np.where(base_try & (cldtopk < cldbasek), NO_LEVEL, cldbasek)
    base_ok = base_try & (cldbasek > 0) & (cldbasek > 1)
    counts.cldbasek_found += int(base_ok.sum())
    zero_levels(base_ok, 1, cldbasek - 1)

    live = ~done
    qvs = np.zeros((nl, ncol), dtype=F32)
    rh = np.zeros((nl, ncol), dtype=F32)
    rhoa = np.zeros((nl, ncol), dtype=F32)
    for k in range(1, kte):
        i = k - 1
        cldfra[i] = np.where(live, F32(0.0), cldfra[i])
        tc = t[i] - F32(273.15)
        qvsw = rslf(p[i], t[i])
        qvsi = np.where(tc < F32(0.0), rsif(p[i], t[i] + F32(0.025)), qvsw)
        mixed = qvsw - (qvsw - qvsi) * (F32(-12.0) - tc) \
            / (F32(-12.0) + F32(35.))
        qvs_k = np.where(tc >= F32(-12.0), qvsw,
                         np.where(tc < F32(-35.0), qvsi, mixed))
        liquid = live & (qc[i] > F32(1.E-8))
        qv[i] = np.where(liquid, np.maximum(qv[i], qvsw), qv[i])
        qvs_k = np.where(liquid, qvsw, qvs_k)
        icy = live & (qc[i] <= F32(1.E-8)) & (qi[i] >= F32(1.E-9))
        qv[i] = np.where(icy, np.maximum(qv[i], qvsi), qv[i])
        qvs_k = np.where(icy, qvsi, qvs_k)
        qvs[i] = qvs_k
        rh[i] = np.maximum(F32(0.01), qv[i] / qvs_k)
        rhoa[i] = p[i] / (F32(287.0) * t[i])

    ocean = (xland - F32(1.5)) > F32(0.0)
    sel = live & imp_any_top & ocean & (cldtopk > 0) & (cldtopk < kte)
    if sel.any():
        counts.ocean_rh_nocloud += int(sel.sum())
        rh_top = _at(rh, cldtopk)
        simple = sel & ((cldbasek < 0) | (cldbasek > cldtopk))
        cldbasek = np.where(simple & (rh_top < RH_NOCLOUD), cldtopk + 1,
                            np.where(simple, cldtopk, cldbasek))
        scan = sel & ~simple
        if scan.any():
            tmp = (cldtopk + 1).astype(F32)       # REAL cldbasek_tmp
            open_ = scan.copy()
            for k in range(int(cldtopk.max()), 0, -1):
                cur = open_ & (k <= cldtopk)
                stop = cur & (rh[k - 1] < RH_NOCLOUD)
                open_ = open_ & ~stop
                tmp = np.where(cur & ~stop, F32(k), tmp)
            raise_b = scan & (tmp > cldbasek.astype(F32))
            cldbasek = np.where(raise_b, tmp.astype(np.int64), cldbasek)

    for k in range(1, kte):
        i = k - 1
        delz = np.maximum(F32(100.), dz[i])
        rh00l = F32(0.65) + np.sqrt(F32(1.) / (
            F32(25.0) + gridkm * gridkm * delz * F32(0.01)))
        rh00o = F32(0.81) + np.sqrt(F32(1.) / (
            F32(50.0) + gridkm * gridkm * delz * F32(0.01)))
        cond = (qc[i] > F32(1.E-7)) | (qi[i] >= F32(1.E-7)) \
            | ((qs[i] > F32(1.E-6)) & (t[i] < F32(273.)))
        rh00 = np.where(ocean, rh00o, rh00l)
        tc = t[i] - F32(273.15)
        rh00 = np.where(tc < F32(-12.0), rh00l, rh00)
        warm_rhum = np.minimum(rh[i], F32(1.0))
        warm = np.maximum(F32(0.), F32(1.0) - np.sqrt(
            (F32(1.005) - warm_rhum) / (F32(1.005) - rh00)))
        cold_rhum = np.minimum(rh[i], F32(1.45))
        cold_rh00 = rh00 + (F32(1.45) - rh00) * (F32(-12.0) - tc) \
            / (F32(-12.0) + F32(100.))
        cold = np.maximum(F32(0.), F32(1.0) - np.sqrt(
            (F32(1.5) - cold_rhum) / (F32(1.5) - cold_rh00)))
        frac = np.where(tc >= F32(25.0), F32(0.0),
                        np.where(tc >= F32(-12.0), warm, cold))
        frac = np.where(frac > F32(0.), np.maximum(
            F32(0.01), np.minimum(frac, F32(0.9))), frac)
        counts.rh_cloud_fraction += int((live & ~cond & (frac > 0)).sum())
        cldfra[i] = np.where(live, np.where(cond, F32(1.0), frac),
                             cldfra[i])

    keep = np.zeros(ncol, dtype=bool)
    sel = live & imp_any_top & (cldtopk > 0) & (cldtopk < kte)
    if sel.any():
        for k in range(1, kte + 1):
            s = sel & (k >= cldtopk + 1)
            cldfra[k - 1] = np.where(s, F32(0.0), cldfra[k - 1])
        keep = sel.copy()
        thresh_k = np.full(ncol, NO_LEVEL, dtype=np.int64)
        open_ = sel.copy()
        for k in range(int(cldtopk.max()), 0, -1):
            cur = open_ & (k <= cldtopk)
            hit = cur & (cldfra[k - 1] > CLDFRA_THRESH)
            thresh_k = np.where(hit, k, thresh_k)
            open_ = open_ & ~hit
        extend = sel & (thresh_k > 0) & (thresh_k < cldtopk)
        counts.cloud_extended_to_top += int(extend.sum())
        for k in range(1, kte):
            s = extend & (k >= thresh_k + 1) & (k <= cldtopk)
            cldfra[k - 1] = np.where(s, CLDFRA_DEF, cldfra[k - 1])

    sel = live & imp_base & (cldbasek > 1) & (cldbasek <= kte)
    if sel.any():
        for k in range(1, kte):
            s = sel & (k <= cldbasek - 1)
            cldfra[k - 1] = np.where(s, F32(0.0), cldfra[k - 1])
        thresh_k = np.full(ncol, NO_LEVEL, dtype=np.int64)
        open_ = sel.copy()
        for k in range(1, kte + 1):
            cur = open_ & (k >= cldbasek)
            hit = cur & (cldfra[k - 1] > CLDFRA_THRESH)
            thresh_k = np.where(hit, k, thresh_k)
            open_ = open_ & ~hit
        add = sel & (thresh_k > cldbasek)
        for k in range(1, kte):
            s = add & (k >= cldbasek) & (k <= thresh_k - 1) \
                & (rh[k - 1] > RH_NOCLOUD)
            counts.cloud_added_above_base += int(s.sum())
            cldfra[k - 1] = np.where(s, CLDFRA_DEF, cldfra[k - 1])

    if live.any():
        idx = np.flatnonzero(live)
        sub = lambda a: a[:, idx]  # noqa: E731
        cfr, cqc, cqi, cqs = (sub(a) for a in (cldfra[:nl], qc, qi, qs))
        find_cloud_layers(sub(qvs), cfr, sub(t), sub(p), sub(dz), cqc, cqi,
                          cqs, keep[idx], counts)
        _adjust_cloud_final(cfr, cqc, cqi, sub(rhoa), sub(dz), counts)
        cldfra[:nl, idx] = cfr
        qc[:, idx] = cqc
        qi[:, idx] = cqi
        for k in range(1, kte):
            c = cldfra[k - 1]
            s = live & (c > F32(0.20)) & (c < F32(1.0))
            counts.qv_raised += int((s & (qvs[k - 1] > qv[k - 1])).sum())
            qv[k - 1] = np.where(s, np.maximum(qv[k - 1], qvs[k - 1]),
                                 qv[k - 1])


def _adjust_cloud_final(cfr, qc, qi, rho, dz, counts):
    """module_madwrf.F:955-992 adjust_cloudFinal."""
    nl, ncol = cfr.shape
    lwp = np.zeros(ncol, dtype=F32)
    iwp = np.zeros(ncol, dtype=F32)
    for k in range(nl):
        frac = (cfr[k] > F32(0.0)) & (cfr[k] < F32(1.0))
        lwp = np.where(frac, lwp + qc[k] * rho[k] * dz[k], lwp)
        iwp = np.where(frac, iwp + qi[k] * rho[k] * dz[k], iwp)
    for path, q, key in ((lwp, qc, "lwp_capped"), (iwp, qi, "iwp_capped")):
        cap = path > F32(1.0)
        if not cap.any():
            continue
        setattr(counts, key, getattr(counts, key) + int(cap.sum()))
        xfac = F32(1.0) / np.where(cap, path, F32(1.0))
        for k in range(nl):
            frac = cap & (cfr[k] > F32(0.0)) & (cfr[k] < F32(1.0))
            q[k] = np.where(frac, q[k] * xfac, q[k])


def insert_clouds_mode(flag_cldmask, flag_cldtopz, flag_cldbasez,
                       flag_brtemp) -> int:
    """module_madwrf.F:104-107: the cloud initialization method."""
    mode = 0
    if flag_cldmask == 1 and flag_cldtopz == 0 and flag_brtemp == 0:
        mode = 1
    if (flag_cldmask == 1 and flag_brtemp == 1) or flag_cldtopz == 1:
        mode = 2
    if mode == 2 and flag_cldbasez == 1:
        mode = 3
    return mode


def skip_middle_points(ny, nx, em_width, hold_ups):
    """share/module_soil_pre.F:3084-3110 over the (ny, nx) mass grid.

    WRF's ide/jde are one past the last mass point, so ``ide-1-width`` is
    ``nx-1-width`` in 1-based i; a 1-row/1-column domain (ide or jde of 3
    in WRF terms means two mass points) never skips."""
    skip = np.zeros((ny, nx), dtype=bool)
    if hold_ups or nx + 1 == 3 or ny + 1 == 3:
        return skip
    jj, ii = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1),
                         indexing="ij")
    return ((ii >= 1 + em_width) & (ii <= nx - em_width)
            & (jj >= 1 + em_width) & (jj <= ny - em_width))


def init_madwrf_clouds(*, t, p, ph_2, phb, qv, qc, qi, qs, xland, ht,
                       cldmask, cldtopz, cldbasez, brtemp, dx, dy, p00,
                       flag_cldmask=0, flag_cldtopz=0, flag_cldbasez=0,
                       flag_brtemp=0, em_width=1, hold_ups=True,
                       cldfra=None, counts=None):
    """module_madwrf.F:71-204 Init_madwrf_clouds over (k, j, i) fields.

    Mass fields are (nz, ny, nx) FP32 and are updated IN PLACE (qv, qc,
    qi, qs; cldmask/cldtopz are INOUT as in WRF); ph_2/phb are (nz+1, ny,
    nx).  ``t`` is the dry potential-temperature perturbation (WRF's t_2
    before the theta_m conversion at module_initialize_real.F:4904) and
    ``p`` the full pressure (WRF's p_hyd = p + pb).  Returns ``cldfra``
    with nz+1 levels (WRF's kte level included); levels the routine does
    not write keep the values of the ``cldfra`` passed in (zeros if None).
    """
    counts = ColumnCounts() if counts is None else counts
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _init_madwrf_clouds(
            t=t, p=p, ph_2=ph_2, phb=phb, qv=qv, qc=qc, qi=qi, qs=qs,
            xland=xland, ht=ht, cldmask=cldmask, cldtopz=cldtopz,
            cldbasez=cldbasez, brtemp=brtemp, dx=dx, dy=dy, p00=p00,
            flag_cldmask=flag_cldmask, flag_cldtopz=flag_cldtopz,
            flag_cldbasez=flag_cldbasez, flag_brtemp=flag_brtemp,
            em_width=em_width, hold_ups=hold_ups, cldfra=cldfra,
            counts=counts)


def _init_madwrf_clouds(*, t, p, ph_2, phb, qv, qc, qi, qs, xland, ht,
                        cldmask, cldtopz, cldbasez, brtemp, dx, dy, p00,
                        flag_cldmask, flag_cldtopz, flag_cldbasez,
                        flag_brtemp, em_width, hold_ups, cldfra, counts):
    """The body of :func:`init_madwrf_clouds` (masked lanes compute values
    that are then discarded, which is what the errstate silences)."""
    nz, ny, nx = t.shape
    if cldfra is None:
        cldfra = np.zeros((nz + 1, ny, nx), dtype=F32)
    mode = insert_clouds_mode(flag_cldmask, flag_cldtopz, flag_cldbasez,
                              flag_brtemp)
    gridkm = np.sqrt(F32(dx) * F32(dy)) * F32(0.001)
    run = ~skip_middle_points(ny, nx, em_width, hold_ups)
    jj, ii = np.nonzero(run)
    if jj.size == 0:
        return cldfra, counts

    def col(a):
        return np.ascontiguousarray(a[:, jj, ii], dtype=F32)

    tt, pp = col(t), col(p)
    ts = (tt + T0) / _powf(F32(p00) / pp, RCP)
    geo = col(ph_2) + col(phb)
    dzs = (geo[1:] - geo[:-1]) / G
    # The two sums are WRF's ((ph_2+phb)(k+1) - (ph_2+phb)(k)) / G: the
    # parenthesised additions round first, exactly as above.
    ht_c = ht[jj, ii].astype(F32)
    _, tropoz = calc_tropo_height(ts, pp, dzs)
    tropoz = tropoz + ht_c
    cm = cldmask[jj, ii].astype(F32)
    ctz = cldtopz[jj, ii].astype(F32)
    agl = np.full(jj.size, MISSING, dtype=F32)
    if mode > 0:
        if flag_cldtopz == 1:
            cm = np.where(ctz > F32(0.0), F32(1.0),
                          np.where(ctz == F32(0.0), F32(0.0), MISSING))
            agl = np.where(ctz > F32(0.0),
                           np.maximum(F32(0.0), ctz - ht_c), agl)
        elif flag_brtemp == 1:
            ctz = calc_cldtopz_from_brtemp(cm, ts, dzs,
                                           brtemp[jj, ii].astype(F32),
                                           tropoz, ht_c)
            agl = np.where(ctz > F32(0.0),
                           np.maximum(F32(0.0), ctz - ht_c), agl)
        cldmask[jj, ii] = cm
        cldtopz[jj, ii] = ctz
    cf = np.ascontiguousarray(cldfra[:, jj, ii], dtype=F32)
    cqv, cqc, cqi, cqs = col(qv), col(qc), col(qi), col(qs)
    cal_cldfra3_madwrf(cf, cqv, cqc, cqi, cqs, dzs, pp, ts,
                       xland[jj, ii].astype(F32), gridkm,
                       np.full(jj.size, mode), cm, agl,
                       cldbasez[jj, ii].astype(F32), counts)
    cldfra[:, jj, ii] = cf
    qv[:, jj, ii] = cqv
    qc[:, jj, ii] = cqc
    qi[:, jj, ii] = cqi
    qs[:, jj, ii] = cqs
    return cldfra, counts


def cloud_tracer_nudge_np(qc, qi, qs, tr_qc, tr_qi, tr_qs, dt, dt_relax,
                          dt_nudge, xtime):
    """dyn_em/module_big_step_utilities_em.F:6711-6756, in place.

    ``q = q + (tr - q) * dt / dt_relax`` evaluated left to right in FP32,
    only while ``xtime < dt_nudge`` (both in minutes)."""
    if not F32(xtime) < F32(dt_nudge):
        return False
    dt, dt_relax = F32(dt), F32(dt_relax)
    for q, tr in ((qc, tr_qc), (qi, tr_qi), (qs, tr_qs)):
        q[...] = q + (tr - q) * dt / dt_relax
    return True


def minutes_since_start(seconds_since_start) -> np.float32:
    """frame/module_domain.F:2177-2191 minutesSinceSimulationStart.

    ``REAL(days)*24.*60. + REAL(seconds)/60.`` plus the fractional-second
    term ``(REAL(Sn)/REAL(Sd))/60.`` for a non-integer clock."""
    from fractions import Fraction
    whole = Fraction(seconds_since_start)
    days, rem = divmod(whole, 86400)
    secs = int(rem)
    frac = rem - secs
    value = F32(days) * F32(24.) * F32(60.) + F32(secs) / F32(60.)
    if frac:
        value = value + (F32(frac.numerator) / F32(frac.denominator)) \
            / F32(60.)
    return F32(value)


# --------------------------------------------------------------------------
# Device path: kernels/madwrf.cu on the no-FTZ, no-FMA route.
# --------------------------------------------------------------------------

#: The direct-NVRTC compile options.  RawModule appends ``-ftz=true`` after
#: caller options (cupy/cuda/compiler.py), which would flush the subnormal
#: hydrometeor words WRF's nudge keeps; this unit therefore compiles through
#: ``compile_using_nvrtc`` itself, the woof.ensemble.recentered route.
CUDA_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.kernels:madwrf"
_TPB = 128
#: Columns per cloud-initialization launch.  The six FP32 intermediates
#: (ts, dzs, theta, qvs, rh, rhoa) are a (6, nz, columns) scratch; the
#: chunk keeps it near 64 MiB at 50 levels whatever the domain size.
CLDINIT_CHUNK_COLUMNS = 1 << 16


@cuda_cache(maxsize=None)
def _cuda_module():
    """kernels/madwrf.cu, compiled once per device on the direct route."""
    from woof.core.kernels import load_module_no_ftz
    return load_module_no_ftz("madwrf", no_fmad=True)


def _kernel(name):
    return _cuda_module().get_function(name)


def init_madwrf_clouds_device(*, t, p, ph_2, phb, qv, qc, qi, qs, xland, ht,
                              cldmask, cldtopz, cldbasez, brtemp, dx, dy,
                              p00, flag_cldmask=0, flag_cldtopz=0,
                              flag_cldbasez=0, flag_brtemp=0, em_width=1,
                              hold_ups=True, cldfra=None):
    """:func:`init_madwrf_clouds` on the card, same arguments, CuPy arrays.

    Every array must be C-contiguous FP32 on the current device; qv/qc/qi/
    qs, cldfra, cldmask and cldtopz are updated in place."""
    import cupy as cp
    nz, ny, nx = t.shape
    if cldfra is None:
        cldfra = cp.zeros((nz + 1, ny, nx), dtype=F32)
    arrays = dict(t=t, p=p, ph_2=ph_2, phb=phb, qv=qv, qc=qc, qi=qi, qs=qs,
                  cldfra=cldfra, xland=xland, ht=ht, cldmask=cldmask,
                  cldtopz=cldtopz, cldbasez=cldbasez, brtemp=brtemp)
    for name, arr in arrays.items():
        if arr.dtype != F32 or not arr.flags.c_contiguous:
            raise ValueError(f"MAD-WRF cloud initialization needs C-"
                             f"contiguous float32 {name}, got {arr.dtype}")
    mode = insert_clouds_mode(flag_cldmask, flag_cldtopz, flag_cldbasez,
                              flag_brtemp)
    rows = max(1, min(ny, CLDINIT_CHUNK_COLUMNS // max(nx, 1)))
    scratch = cp.empty((6, nz, rows * nx), dtype=F32)
    status = cp.zeros(1, dtype=cp.int32)
    kern = _kernel("madwrf_cldinit")
    for j0 in range(0, ny, rows):
        nrows = min(rows, ny - j0)
        ncol = nrows * nx
        kern(((ncol + _TPB - 1) // _TPB,), (_TPB,),
             (t, p, ph_2, phb, qv, qc, qi, qs, cldfra, xland, ht, cldmask,
              cldtopz, cldbasez, brtemp, scratch, F32(dx), F32(dy),
              F32(p00), np.int32(flag_cldmask), np.int32(flag_cldtopz),
              np.int32(flag_brtemp), np.int32(mode), np.int32(em_width),
              np.int32(bool(hold_ups)), np.int32(nz), np.int32(ny),
              np.int32(nx), np.int32(j0), np.int32(nrows), status))
    if int(status.get()[0]):
        raise ValueError(
            "MAD-WRF cloud initialization: a column's -40 C or -12 C level "
            "lies within two/three layers of the model top, where WRF's "
            "find_cloudLayers stops the run (module_madwrf.F:673-681, "
            "'problem in temperature profile'): the layer scans above it "
            "would index past the column. Raise the model top or check the "
            "analysis temperature profile.")
    return cldfra


def cloud_tracer_nudge_device(qc, qi, qs, tr_qc, tr_qi, tr_qs, dt, dt_relax,
                              dt_nudge, xtime, *, spec_zone=0, periodic_x=False):
    """:func:`cloud_tracer_nudge_np` on the card (CuPy arrays, in place)."""
    if not F32(xtime) < F32(dt_nudge):
        return False
    n = int(qc.size)
    for arr in (qc, qi, qs, tr_qc, tr_qi, tr_qs):
        if arr.dtype != F32 or not arr.flags.c_contiguous or arr.size != n:
            raise ValueError("MAD-WRF nudging needs six C-contiguous float32 "
                             "fields of one shape")
    _kernel("madwrf_nudge")(((n + 255) // 256,), (256,),
                            (qc, qi, qs, tr_qc, tr_qi, tr_qs, F32(dt),
                             F32(dt_relax), np.int64(n), np.int32(qc.shape[-1]),
                             np.int32(qc.shape[-2]), np.int32(spec_zone), np.int32(periodic_x)))
    return True
