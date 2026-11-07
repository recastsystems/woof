"""Device cloud fraction for FLG, copied unchanged from the shared WRF driver port.

This boundary does not load RRTMGP coefficients. WRF v4.6.1
module_radiation_driver.F:3761-3986 is the public-domain authority.
"""

import numpy as np

DTYPE = np.float32

def _host_expf(x):
    """WOOF's own float32 expf over a host array (noahmp_libm, bitwise the
    device slab_expf)."""
    from woof.core.noahmp_libm import expf
    x = np.asarray(x, np.float32)
    out = np.empty(x.shape, np.float32)
    flat = out.reshape(-1)
    for i, v in enumerate(x.reshape(-1)):
        flat[i] = expf(v)
    return out

def _host_powf(x, y):
    """WOOF's own float32 powf over host arrays (noahmp_libm.powf_array)."""
    from woof.core.noahmp_libm import powf_array
    return np.asarray(powf_array(np.asarray(x, np.float32),
                                 np.asarray(y, np.float32)), np.float32)

def cal_cldfra1(qv, qc, qi, qs, tlay, play, *, f_qc=True, f_qi=True,
                f_qs=True, mp_physics=None, f_ice_phy=None):
    """WRF icloud=1 Xu-Randall cloud fraction on device (FP32).

    Exact transcription of WRF v4.6.1 ``module_radiation_driver.F``
    ``cal_cldfra1`` (lines 3761-3986), the routine the radiation driver
    calls for the Registry default ``icloud=1`` (driver lines 1320-1332,
    Registry.EM_COMMON:2498).  Saturation follows Murray (1966) with the
    driver's constants (lines 3806-3816, 3861-3865); the liquid/ice
    saturation blend uses the condensate ice weight (line 3945) and the
    fraction is Xu and Randall (1996) with ALPHA0=100, GAMMA=0.49,
    QCLDMIN=1e-12, PEXP=0.25, RHGRID=1.0 plus the -6.9 ARG clamp and the
    0.01 truncation (lines 3950-3979).

    Moisture-set dispatch mirrors the driver flags: ``f_qc and f_qi and
    f_qs`` is the Morrison-class branch (lines 3870-3877, QCLD=QI+QC+QS,
    weight=(QI+QS)/QCLD); ``f_qc and f_qi and not f_qs`` is the branch WRF
    comments "for P3, mp option 50 or 51" (lines 3879-3887, QCLD=QI+QC,
    weight=QI/QCLD), for a package with one ice category and no snow
    species; ``f_qc`` alone is the Kessler-class branch (lines 3891-3899,
    QCLD=QC, 273.15 K phase threshold).  Rain never enters QCLD (lines
    3904-3916).

    ``mp_physics`` 5 or 15 (Ferrier-Aligo, fer_mp_hires and its advected
    sibling) adds WRF's own selector arm at lines 3926-3937, which runs
    AFTER the P3 arm above for the same qc+qi/no-qs package: QCLD is
    QC+QI, the weight QI/QCLD, and below -40 C the weight is 1 (all
    condensate ice at homogeneous freezing, the scheme's T_ICE).
    """
    if f_qc and f_qs and not f_qi:
        from woof.core.rrtmgp import cal_cldfra1 as shared_cloud_fraction
        return shared_cloud_fraction(qv, qc, qi, qs, tlay, play,
            f_qc=f_qc, f_qi=f_qi, f_qs=f_qs, mp_physics=mp_physics,
            f_ice_phy=f_ice_phy)
    import cupy as cp

    qv = cp.ascontiguousarray(cp.asarray(qv, dtype=DTYPE))
    if qv.ndim != 2:
        raise ValueError("qv must have shape (ncol,nlay)")
    qc = _device_profile(qc, qv.shape, "qc")
    qi = _device_profile(qi, qv.shape, "qi")
    qs = _device_profile(qs, qv.shape, "qs")
    tlay = _device_profile(tlay, qv.shape, "tlay")
    play = _device_profile(play, qv.shape, "play")
    if not f_qc:
        raise NotImplementedError(
            "cal_cldfra1 requires a cloud-water species (F_QC)")
    # EXP and ** on REAL(4) are WOOF's own float32 expf and powf
    # (woof.core.noahmp_slab_libm), not CuPy's CUDA builtins: cp.exp and
    # cp.power round differently and moved CLDFRA by up to 2,113 ULP on the
    # legacy RRTMG column oracle (tools/rrtmg_legacy_column_oracle).  Host
    # arrays (the CPU-emulated adapter tests substitute NumPy for CuPy) take
    # the same functions' host twins, bit for bit.
    if type(qv).__module__.split(".")[0] == "cupy":
        from woof.core.noahmp_slab_libm import slab_expf, slab_powf
    else:
        slab_expf, slab_powf = _host_expf, _host_powf
    qcldmin = DTYPE(1.0e-12)
    svpt0 = DTYPE(273.15)
    tc = tlay - svpt0
    esw = DTYPE(1000.0) * DTYPE(0.61078) * slab_expf(
        DTYPE(17.2693882) * tc / (tlay - DTYPE(35.86)))
    esi = DTYPE(1000.0) * DTYPE(0.61078) * slab_expf(
        DTYPE(21.8745584) * tc / (tlay - DTYPE(7.66)))
    ep2 = DTYPE(287.0) / DTYPE(461.6)
    qvsw = ep2 * esw / (play - esw)
    qvsi = ep2 * esi / (play - esi)
    if f_qi and f_qs:
        qcld = qi + qc + qs
        weight = cp.where(qcld < qcldmin, DTYPE(0.0),
                          (qi + qs) / cp.maximum(qcld, qcldmin))
    elif f_qi:
        # module_radiation_driver.F:3879-3887, "for P3, mp option 50 or
        # 51": one ice category, no snow species, so QS never enters QCLD
        # and the ice weight is QI/QCLD.  qs is ignored here rather than
        # added as a zero -- a package without qs has no such array.
        qcld = qi + qc
        weight = cp.where(qcld < qcldmin, DTYPE(0.0),
                          qi / cp.maximum(qcld, qcldmin))
        if mp_physics is not None and int(mp_physics) in (5, 15):
            # module_radiation_driver.F:3926-3937, FER_MP_HIRES and
            # FER_MP_HIRES_ADVECT: QCLD = QC + QI (the same sum), weight
            # QI/QCLD, and weight = 1 where tc < -40.
            weight = cp.where(
                qcld < qcldmin, DTYPE(0.0),
                cp.where(tc < DTYPE(-40.0), DTYPE(1.0), weight))
    else:
        qcld = qc
        weight = cp.where(qcld < qcldmin, DTYPE(0.0),
                          cp.where(tlay > svpt0, DTYPE(0.0), DTYPE(1.0)))
    qvs_weight = (DTYPE(1.0) - weight) * qvsw + weight * qvsi
    rhum = qv / qvs_weight
    subsat = cp.maximum(DTYPE(1.0e-10), qvs_weight - qv)
    arg = cp.maximum(DTYPE(-6.9),
                     DTYPE(-100.0) * qcld / slab_powf(subsat, DTYPE(0.49)))
    fraction = slab_powf(cp.maximum(DTYPE(1.0e-10), rhum), DTYPE(0.25)) \
        * (DTYPE(1.0) - slab_expf(arg))
    fraction = cp.where(fraction < DTYPE(0.01), DTYPE(0.0), fraction)
    return cp.ascontiguousarray(
        cp.where(qcld < qcldmin, DTYPE(0.0),
                 cp.where(rhum >= DTYPE(1.0), DTYPE(1.0), fraction)))

def _device_profile(value, shape, name):
    import cupy as cp
    out = cp.ascontiguousarray(cp.asarray(value, dtype=DTYPE))
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    return out
