# ======================================================================
# THIRD-PARTY NOTICE.  Parts of this file are hand transcriptions of
# third-party work.  ArWen distributes the file under the Apache License
# 2.0; the notices below belong to the transcribed parts and are kept here
# because their own licences require it.  Full texts are in the repository
# NOTICE and in the licenses/ directory.
#
# For the two libm grants the text also sits beside the code, in
# woof/core/kernels/LICENSE-third-party.txt.
#
#   FDLIBM -- expm1f and tanhf. Developed at SunPro and converted to
#   single precision at Cygnus Support; glibc carries it substantially
#   unmodified and presents it, in its own LICENSES file, as Sun's code.
#   The notice below is the whole of the licence: its one condition is
#   that it be preserved.
#
#       Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#
#       Developed at SunPro, a Sun Microsystems, Inc. business.
#       Permission to use, copy, modify, and distribute this
#       software is freely granted, provided that this notice
#       is preserved.
# ======================================================================
"""CPU references for pinned WRF v4.6.1 MYNN PBL column routines.

The first routine transcribed is ``module_bl_mynn.F:mym_level2``.  It operates
on independent adjacent-level pairs, which is exactly how each output
interface is computed.  This numerical reference does not admit the coupled
MYNN runtime selector.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# glibc 2.39 transcriptions.  The module name is the Noah-MP lane's; the
# functions are not Noah-MP-specific, and its own docstring says so.  They are
# imported rather than re-derived because rounding an FP64 evaluation is a
# different function from what the oracle's libm computes.  Import directly
# under the reference's private names rather than adding delegation wrappers.
# The expf FP64-then-round shim disagreed on 0.062% of a 35-million-point
# sweep: mym_initialize's (b1*pmz*elv)**(2/3) on a 30-level cloudy column
# moved el by 25,193 ULP downstream.  Real-exponent ** uses powf; that cold
# start failed before it too was repointed to glibc.  FP32 ATAN uses atanf:
# glibc is faithfully, not correctly, rounded; NumPy's FP32 atan is platform
# dependent and FP64-then-round gives a third function.  The stfunc oracle's
# (1-phi_m)/zet cancellation amplified that error: 22/814 phim and 9/814 phih
# values differed by up to 80/84 ULP; all four counts are zero with glibc.
# log10f and expm1f likewise use the shared glibc 2.39 FP32 transcriptions.
from woof.core.noahmp_libm import (
    atanf as _atanf,
    expf as _expf,
    expm1f as _expm1f,
    log10f as _log10f,
    logf as _glibc_logf,
    powf as _powf,
)


F = np.float32
PR = F(0.74)
G1 = F(0.235)
B1 = F(24.0)
B2 = F(15.0)
C2 = F(0.729)
C3 = F(0.340)
C5 = F(0.2)
A1 = F(B1 * F(F(1.0) - F(3.0) * G1) / F(6.0))
C1 = F(G1 - F(1.0) / (F(3.0) * A1 * F(2.88449914061481660)))
A2 = F(A1 * F(G1 - C1) / F(G1 * PR))
G2 = F(B2 / B1 * F(F(1.0) - C3)
       + F(2.0) * A1 / B1 * F(F(3.0) - F(2.0) * C2))
TV0 = F(F(F(461.6) / F(287.0) - F(1.0)) * F(300.0))
GTR = F(F(9.81) / F(300.0))
CC2 = F(F(1.0) - C2)
CC3 = F(F(1.0) - C3)
E1C = F(F(3.0) * A2 * B2 * CC3)
E2C = F(F(9.0) * A1 * A2 * CC2)
E3C = F(F(9.0) * A2 * A2 * CC2 * F(F(1.0) - C5))
E4C = F(F(12.0) * A1 * A2 * CC2)
E5C = F(F(6.0) * A1 * A1)
# ``3.0*c1*e5c*gmel`` at module_bl_mynn.F:2845 associates left to right, and
# kind_phys is kind(1.0) here, so ``3.0*c1`` and then ``*e5c`` are FP32
# roundings before the DOUBLE PRECISION ``gmel`` widens the product.  Doing the
# same three factors in FP64 is a different number -- 3.435353333994499 rather
# than 3.4353532791137695 -- and it was worth one ULP of ``sh`` wherever the
# level-2.5 branch ran.  Fold it once, at the Fortran's precision.
THREE_C1 = F(F(3.0) * C1)
THREE_C1_E5C = F(THREE_C1 * E5C)
# onethird and twothirds are module_bl_mynn_common.F:68-69; karman = 0.4 is
# module_model_constants.F:82, reaching MYNN through the use statement at
# module_bl_mynn_common.F:25; qkemin = 1.e-3 is module_bl_mynn.F:306.  pmz and
# phh are mym_initialize local initialisers (module_bl_mynn.F:1545): both are
# 1, so b1*pmz and phh*b2 fold to b1 and b2 but are kept named so the
# transcription reads like the Fortran.
ONETHIRD = F(F(1.0) / F(3.0))
TWOTHIRDS = F(F(2.0) / F(3.0))
KARMAN = F(0.4)
QKEMIN = F(1.0e-3)
PMZ = F(1.0)
PHH = F(1.0)
B1_PMZ = F(B1 * PMZ)
PHH_B2 = F(PHH * B2)
# module_bl_mynn.F:1595 fixes the mym_initialize iteration count.
MYM_INITIALIZE_ITERATIONS = 5
EXPM1F_TINY = F(1.0e-30)

# module_model_constants.F / module_bl_mynn_common.F thermodynamic constants,
# rounded exactly the way the Fortran parameter chain rounds them.
RD = F(287.0)
RV = F(461.6)
CP = F(F(7.0) * RD / F(2.0))
CPV = F(F(4.0) * RV)
CLIQ = F(4190.0)
CICE = F(2106.0)
XLV = F(2.5e6)
XLF = F(3.50e5)
XLS = F(2.85e6)
EP_2 = F(RD / RV)
XLVCP = F(XLV / CP)
# module_bl_mynn_common.F:86 derives xlscp from (xlv+xlf), not from xls.
XLSCP = F(F(XLV + XLF) / CP)
P608 = F(F(RV / RD) - F(1.0))
T0C = F(273.15)
TICE = F(240.0)
TLIQ = F(269.0)
T0C_M6 = F(T0C - F(6.0))
CPV_CLIQ = F(CPV - CLIQ)
CPV_CICE = F(CPV - CICE)
# mym_condensation CASE(2) parameters.
QPCT_SFC = F(0.025)
QPCT_PBL = F(0.030)
QPCT_TRP = F(0.040)
RHCRIT = F(0.83)
RHMAX = F(1.02)
EXP_M1 = F(np.exp(-1.0))
TROPO_LAPSE = F(F(10.0) / F(1500.0))
# esat_blend / qsat_blend saturation-vapour-pressure polynomials (Pa).
_ESAT_LIQUID = (
    F(0.611583699e03), F(0.444606896e02), F(0.143177157e01),
    F(0.264224321e-1), F(0.299291081e-3), F(0.203154182e-5),
    F(0.702620698e-8), F(0.379534310e-11), F(-0.321582393e-13),
)
_ESAT_ICE = (
    F(0.609868993e03), F(0.499320233e02), F(0.184672631e01),
    F(0.402737184e-1), F(0.565392987e-3), F(0.521693933e-5),
    F(0.307839583e-7), F(0.105785160e-9), F(0.161444444e-12),
)

MYNN_LEVEL2_INPUTS = (
    "dz", "u", "v", "thl", "thetav", "qw", "ql", "vt", "vq",
    "dz_prev", "u_prev", "v_prev", "thl_prev", "thetav_prev",
    "qw_prev", "ql_prev", "vt_prev", "vq_prev",
)
MYNN_LEVEL2_OUTPUTS = ("dtl", "dqw", "dtv", "gm", "gh", "sm", "sh")
MYNN_MIXLENGTH_INPUTS = (
    "dz", "zw", "u", "v", "qke", "dtv", "theta", "vt", "vq",
    "cldfra", "edmf_w", "edmf_a", "xland", "dx", "rmo", "flt",
    "fltv", "flq", "zi", "psig_bl",
)
MYNN_INITIALIZE_COLUMN_INPUTS = (
    "dz", "u", "v", "thl", "qw", "theta", "thetav", "cldfra",
    "edmf_w", "edmf_a", "sm", "sh", "qke",
)
MYNN_INITIALIZE_SCALAR_INPUTS = (
    "xland", "dx", "rmo", "ust", "zi", "psig_bl",
)
MYNN_INITIALIZE_INPUTS = (
    *MYNN_INITIALIZE_COLUMN_INPUTS, "zw", *MYNN_INITIALIZE_SCALAR_INPUTS,
)
MYNN_INITIALIZE_OUTPUTS = ("el", "qke", "tsq", "qsq", "cov", "sm", "sh")
MYNN_TURBULENCE_INPUTS = (
    "dz", "zw", "u", "v", "thl", "thetav", "ql", "qw", "qke",
    "tsq", "qsq", "cov", "vt", "vq", "theta", "cldfra", "edmf_w",
    "edmf_a", "tkeprodtd", "xland", "dx", "rmo", "flt", "fltv",
    "flq", "zi", "psig_bl", "psig_shcu",
)
MYNN_PREDICT_INPUTS = (
    "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el", "s_aw",
    "s_awqke", "ust", "flt", "flq", "pmz", "phh", "delt", "qke",
    "tsq", "qsq", "cov",
)
MYNN_CONDENSATION_INPUTS = (
    "dz", "zw", "th", "thl", "qw", "qv", "qc", "qi", "qs", "p",
    "exner", "tsq", "qsq", "cov", "sh", "el", "rstoch", "vt", "vq",
    "sgm", "xland", "dx", "pblh", "hfx", "rmo",
)
MYNN_CONDENSATION_OUTPUTS = (
    "qc_bl", "qi_bl", "cldfra", "vt", "vq", "sgm",
)
# mynn_tendencies argument groups under the mass-flux-free identity.  WRF
# declares qc, qi, qs, qnc, qni, sqw, qnwfa, qnifa, qnbca, tsq, qsq, cov,
# cldfra_bl1d, dfq and flq as well, but none of them is read once
# bl_mynn_mixqt=0 and bl_mynn_mixscalars=0, so they are not required here.
MYNN_TENDENCIES_LAYER_INPUTS = (
    "dz", "rho", "u", "v", "th", "tk", "qv", "p", "exner", "thl",
    "sqv", "sqc", "sqi", "sqs", "ozone", "tcd", "qcd", "dfm", "dfh",
    "diss_heat", "sub_thl", "sub_sqv", "sub_u", "sub_v",
    "det_thl", "det_sqv", "det_sqc", "det_u", "det_v",
)
MYNN_TENDENCIES_INTERFACE_INPUTS = (
    "s_aw", "s_awthl", "s_awqv", "s_awqc", "s_awu", "s_awv",
    "sd_aw", "sd_awthl", "sd_awqv", "sd_awqc", "sd_awu", "sd_awv",
)
MYNN_TENDENCIES_SCALAR_INPUTS = (
    "delt", "psfc", "ust", "wspd", "uoce", "voce", "flt", "flqv", "flqc",
)
# W4 mixscalars admission: the qn-family inputs, required only when
# bl_mynn_mixscalars=1 (module_bl_mynn.F:4654-4860 solves; the s_awqn*
# fluxes come from DMP_mf's scalar_opt>0 accumulation at :6447-6456).
MYNN_TENDENCIES_QN_LAYER_INPUTS = ("qnc", "qni", "qnwfa", "qnifa", "qnbca")
MYNN_TENDENCIES_QN_INTERFACE_INPUTS = (
    "s_awqnc", "s_awqni", "s_awqnwfa", "s_awqnifa", "s_awqnbca",
)
MYNN_TENDENCIES_INPUTS = (
    *MYNN_TENDENCIES_LAYER_INPUTS,
    *MYNN_TENDENCIES_INTERFACE_INPUTS,
    *MYNN_TENDENCIES_SCALAR_INPUTS,
)
MYNN_TENDENCIES_OUTPUTS = (
    "du", "dv", "dth", "dqv", "dqc", "dqi", "dqs", "dqnc", "dqni",
    "dqnwfa", "dqnifa", "dqnbca", "dozone", "thl",
)
# moisture_check floors (module_bl_mynn.F:5162-5164).
QVMIN = F(1.0e-20)
QCMIN = F(0.0)
QIMIN = F(0.0)


def _as_pairs(values: Mapping[str, object]) -> tuple[dict[str, np.ndarray], int]:
    missing = [name for name in MYNN_LEVEL2_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN level-2 inputs: {', '.join(missing)}")
    pairs = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in MYNN_LEVEL2_INPUTS
    }
    shapes = {value.shape for value in pairs.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 1:
        raise ValueError("MYNN level-2 inputs must be equal-length 1-D arrays")
    if any(not np.isfinite(value).all() for value in pairs.values()):
        raise ValueError("MYNN level-2 inputs must be finite")
    if np.any(pairs["dz"] <= 0.0) or np.any(pairs["dz_prev"] <= 0.0):
        raise ValueError("MYNN level-2 layer depths must be positive")
    return pairs, next(iter(shapes))[0]


def mynn_level2_pairs(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Evaluate WRF ``mym_level2`` for independent adjacent-level pairs."""

    source, count = _as_pairs(values)
    result = {
        name: np.empty(count, dtype=np.float32) for name in MYNN_LEVEL2_OUTPUTS
    }
    for i in range(count):
        dz = F(source["dz"][i])
        dz_prev = F(source["dz_prev"][i])
        dz_sum = F(dz + dz_prev)
        dzk = F(F(0.5) * dz_sum)
        afk = F(dz / dz_sum)
        abk = F(F(1.0) - afk)
        du = F(F(source["u"][i]) - F(source["u_prev"][i]))
        dv = F(F(source["v"][i]) - F(source["v_prev"][i]))
        duz = F(F(du * du) + F(dv * dv))
        duz = F(duz / F(dzk * dzk))
        dtz = F(
            F(F(source["thl"][i]) - F(source["thl_prev"][i])) / dzk
        )
        dqz = F(
            F(F(source["qw"][i]) - F(source["qw_prev"][i])) / dzk
        )
        vtt = F(
            F(1.0) + F(F(source["vt"][i]) * abk)
            + F(F(source["vt_prev"][i]) * afk)
        )
        vqq = F(
            TV0 + F(F(source["vq"][i]) * abk)
            + F(F(source["vq_prev"][i]) * afk)
        )
        dtq = F(F(vtt * dtz) + F(vqq * dqz))
        gh = F(-F(dtq * GTR))
        ri = F(-gh / max(duz, F(1.0e-10)))
        a2fac = F(F(1.0) / F(F(1.0) + max(ri, F(0.0))))

        rfc = F(G1 / F(G1 + G2))
        f1 = F(
            B1 * F(G1 - C1)
            + F(3.0) * A2 * a2fac * F(F(1.0) - C2) * F(F(1.0) - C5)
            + F(2.0) * A1 * F(F(3.0) - F(2.0) * C2)
        )
        f2 = F(
            B1 * F(G1 + G2) - F(3.0) * A1 * F(F(1.0) - C2)
        )
        rf1 = F(B1 * F(G1 - C1) / f1)
        rf2 = F(B1 * G1 / f2)
        smc = F(A1 / F(A2 * a2fac) * f1 / f2)
        shc = F(F(3.0) * F(A2 * a2fac) * F(G1 + G2))
        ri1 = F(F(0.5) / smc)
        ri2 = F(rf1 * smc)
        ri3 = F(F(4.0) * rf2 * smc - F(2.0) * ri2)
        ri4 = F(ri2 * ri2)
        radical = F(ri * ri - ri3 * ri + ri4)
        rf = min(F(ri1 * F(ri + ri2 - F(np.sqrt(radical)))), rfc)
        sh = F(shc * F(rfc - rf) / F(F(1.0) - rf))
        sm = F(smc * F(rf1 - rf) / F(rf2 - rf) * sh)

        result["dtl"][i] = dtz
        result["dqw"][i] = dqz
        result["dtv"][i] = dtq
        result["gm"][i] = duz
        result["gh"][i] = gh
        result["sm"][i] = sm
        result["sh"][i] = sh
    return result


def mynn_get_pblh(
    thetav: object,
    qke: object,
    zw: object,
    dz: object,
    landsea: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate WRF ``GET_PBLH`` for independent complete columns.

    Mass-level arrays use shape ``(ncol, nz)`` and ``zw`` uses
    ``(ncol, nz + 1)``. Returned ``kzi`` retains WRF's one-based index.
    """

    theta = np.asarray(thetav, dtype=np.float32)
    energy = np.asarray(qke, dtype=np.float32)
    depth = np.asarray(dz, dtype=np.float32)
    interface = np.asarray(zw, dtype=np.float32)
    if theta.ndim != 2 or energy.shape != theta.shape or depth.shape != theta.shape:
        raise ValueError("MYNN PBLH mass fields must share shape (ncol,nz)")
    ncol, nz = theta.shape
    if nz < 4 or interface.shape != (ncol, nz + 1):
        raise ValueError("MYNN PBLH zw must have shape (ncol,nz+1), nz >= 4")
    sea = np.asarray(landsea, dtype=np.float32)
    try:
        sea = np.broadcast_to(sea, (ncol,))
    except ValueError as exc:
        raise ValueError("landsea is not broadcastable to the column count") from exc
    if any(not np.isfinite(value).all() for value in (
            theta, energy, depth, interface, sea)):
        raise ValueError("MYNN PBLH inputs must be finite")
    if np.any(depth <= 0.0):
        raise ValueError("MYNN PBLH layer depths must be positive")

    zi_out = np.empty(ncol, dtype=np.float32)
    kzi_out = np.empty(ncol, dtype=np.int32)
    for column in range(ncol):
        kzi = 2
        minthv = F(9.0e9)
        k = 2
        while interface[column, k - 1] <= F(200.0):
            if minthv > theta[column, k - 1]:
                minthv = F(theta[column, k - 1])
            k += 1
            if k > nz:
                raise ValueError("MYNN PBLH column has no interface above 200 m")

        delta = F(1.0 if sea[column] >= F(1.5) else 1.25)
        zi = F(0.0)
        for k in range(2, nz):  # WRF kts+1 .. kte-1.
            current = F(theta[column, k - 1])
            if current >= F(minthv + delta):
                denominator = max(
                    F(current - F(theta[column, k - 2])), F(1.0e-6)
                )
                fraction = min(F((current - F(minthv + delta)) / denominator),
                               F(1.0))
                zi = F(interface[column, k - 1]
                       - depth[column, k - 2] * fraction)
            if k == nz - 1:
                zi = F(interface[column, 1])
            if zi != F(0.0):
                break

        maxqke = max(F(energy[column, 0]), F(0.0))
        tkeeps = max(F(maxqke / F(40.0)), F(0.02))
        pblh_tke = F(0.0)
        for k in range(2, nz):
            qtke = max(F(energy[column, k - 1] / F(2.0)), F(0.0))
            qtkem1 = max(F(energy[column, k - 2] / F(2.0)), F(0.0))
            if qtke <= tkeeps:
                denominator = max(F(qtkem1 - qtke), F(1.0e-6))
                fraction = min(F((tkeeps - qtke) / denominator), F(1.0))
                pblh_tke = F(interface[column, k - 1]
                             - depth[column, k - 2] * fraction)
                pblh_tke = max(pblh_tke, F(interface[column, 1]))
            if k == nz - 1:
                pblh_tke = F(interface[column, 1])
            if pblh_tke != F(0.0):
                break

        pblh_tke = min(pblh_tke, F(zi + F(350.0)))
        pblh_tke = max(pblh_tke, max(F(zi - F(350.0)), F(10.0)))
        weight = F(F(0.5) * _tanhf(F((zi - F(200.0)) / F(400.0)))
                   + F(0.5))
        if maxqke > F(0.05):
            zi = F(pblh_tke * F(F(1.0) - weight) + zi * weight)

        for k in range(2, nz):
            if interface[column, k - 1] >= zi:
                kzi = k - 1
                break
        zi_out[column] = zi
        kzi_out[column] = kzi
    return zi_out, kzi_out


def mynn_scale_aware(dx: object, pblh: object) -> tuple[np.ndarray, np.ndarray]:
    """Translate WRF ``SCALE_AWARE`` for broadcast-compatible values."""

    dx_array, pblh_array = np.broadcast_arrays(
        np.asarray(dx, dtype=np.float32), np.asarray(pblh, dtype=np.float32)
    )
    if not np.isfinite(dx_array).all() or not np.isfinite(pblh_array).all() \
            or np.any(dx_array <= 0.0) or np.any(pblh_array <= 0.0):
        raise ValueError("MYNN scale-aware dx and PBLH must be positive and finite")
    local = np.empty(dx_array.shape, dtype=np.float32)
    nonlocal_mix = np.empty(dx_array.shape, dtype=np.float32)
    for index in np.ndindex(dx_array.shape):
        spacing = F(dx_array[index])
        height = F(pblh_array[index])
        ratio = F(max(F(F(2.5) * spacing), F(10.0))
                  / min(height, F(3000.0)))
        power = _powf(ratio, F(0.667))
        square = F(ratio * ratio)
        value = F((square + F(0.106) * power)
                  / (square + F(0.066) * power + F(0.071)))
        local[index] = min(max(value, F(0.0)), F(1.0))

        ratio = F(max(F(F(2.5) * spacing), F(10.0))
                  / min(F(height + F(500.0)), F(3500.0)))
        power = _powf(ratio, F(0.667))
        square = F(ratio * ratio)
        value = F((square + F(0.145) * power)
                  / (square + F(0.172) * power + F(0.170)))
        nonlocal_mix[index] = min(max(value, F(0.0)), F(1.0))
    return local, nonlocal_mix


def _boulac_length(
    zw: np.ndarray, dz: np.ndarray, qtke: np.ndarray, theta: np.ndarray,
    boulac_down_sign: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct single-column translation of WRF ``boulac_length``.

    ``boulac_down_sign`` is correct-mode ML1: the downward search's
    constant-theta (linear) branch divides by ``beta*(theta(iz) -
    theta(izz))``, the bbb -> 0 limit of its quadratic branch, as NCAR/
    MYNN-EDMF 399f50f (PR 94) does; WRF v4.6.1 has the opposite sign.
    """

    nz = dz.size
    dlu = np.empty(nz, dtype=np.float32)
    dld = np.empty(nz, dtype=np.float32)
    lb1 = np.empty(nz, dtype=np.float32)
    lb2 = np.empty(nz, dtype=np.float32)
    beta = GTR
    for iz in range(nz):
        zup = F(0.0)
        dlu[iz] = F(zw[nz] - zw[iz] - F(dz[iz] * F(0.5)))
        zzz = F(0.0)
        zup_inf = F(0.0)
        if iz < nz - 1:
            izz = iz
            found = False
            while not found:
                if izz < nz - 1:
                    dzt = F(dz[izz])
                    zup = F(zup - F(beta * theta[iz] * dzt))
                    zup = F(zup + F(
                        beta * F(theta[izz + 1] + theta[izz])
                        * dzt * F(0.5)
                    ))
                    zzz = F(zzz + dzt)
                    if qtke[iz] < zup and qtke[iz] >= zup_inf:
                        bbb = F(F(theta[izz + 1] - theta[izz]) / dzt)
                        if bbb != F(0.0):
                            base = F(beta * F(theta[izz] - theta[iz]))
                            radical = max(F(base * base + F(
                                F(2.0) * bbb * beta * F(qtke[iz] - zup_inf)
                            )), F(0.0))
                            tl = F(F(-base + F(np.sqrt(radical))) / bbb / beta)
                        elif theta[izz] != theta[iz]:
                            tl = F(F(qtke[iz] - zup_inf)
                                   / F(beta * F(theta[izz] - theta[iz])))
                        else:
                            tl = F(0.0)
                        dlu[iz] = F(zzz - dzt + tl)
                        found = True
                    zup_inf = zup
                    izz += 1
                else:
                    found = True

        zdo = F(0.0)
        zdo_sup = F(0.0)
        dld[iz] = F(zw[iz])
        zzz = F(0.0)
        if iz > 0:
            izz = iz
            found = False
            while not found:
                if izz > 0:
                    dzt = F(dz[izz - 1])
                    zdo = F(zdo + F(beta * theta[iz] * dzt))
                    zdo = F(zdo - F(
                        beta * F(theta[izz - 1] + theta[izz])
                        * dzt * F(0.5)
                    ))
                    zzz = F(zzz + dzt)
                    if qtke[iz] < zdo and qtke[iz] >= zdo_sup:
                        bbb = F(F(theta[izz] - theta[izz - 1]) / dzt)
                        if bbb != F(0.0):
                            base = F(beta * F(theta[izz] - theta[iz]))
                            radical = max(F(base * base + F(
                                F(2.0) * bbb * beta * F(qtke[iz] - zdo_sup)
                            )), F(0.0))
                            tl = F(F(base + F(np.sqrt(radical))) / bbb / beta)
                        elif theta[izz] != theta[iz]:
                            drop = (F(theta[iz] - theta[izz]) if boulac_down_sign
                                    else F(theta[izz] - theta[iz]))
                            tl = F(F(qtke[iz] - zdo_sup) / F(beta * drop))
                        else:
                            tl = F(0.0)
                        dld[iz] = F(zzz - dzt + tl)
                        found = True
                    zdo_sup = zdo
                    izz -= 1
                else:
                    found = True

        dld[iz] = min(dld[iz], F(zw[iz + 1]))
        lb1[iz] = min(dlu[iz], dld[iz])
        dlu[iz] = max(F(0.1), min(dlu[iz], F(1000.0)))
        dld[iz] = max(F(0.1), min(dld[iz], F(1000.0)))
        lb2[iz] = F(np.sqrt(F(dlu[iz] * dld[iz])))
        lb1[iz] = F(lb1[iz] / F(F(1.0) + lb1[iz] / F(2000.0)))
        lb2[iz] = F(lb2[iz] / F(F(1.0) + lb2[iz] / F(2000.0)))
        if iz == nz - 1:
            lb1[iz] = lb1[iz - 1]
            lb2[iz] = lb2[iz - 1]
    return lb1, lb2


def _mym_length_column(
    dz: np.ndarray,
    zw: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    qke: np.ndarray,
    dtv: np.ndarray,
    theta: np.ndarray,
    edmf_w: np.ndarray,
    edmf_a: np.ndarray,
    rmo: np.float32,
    fltv: np.float32,
    zi: np.float32,
    psig_bl: np.float32,
    boulac_down_sign: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-column WRF ``mym_length`` CASE(1), ``module_bl_mynn.F:1999-2098``.

    ``xland``, ``dx``, ``flt``, ``flq``, ``vt``, ``vq`` and ``cldfra_bl1D``
    reach the Fortran routine but are not arguments here, and they fall into
    two classes.  ``flt``, ``flq``, ``vt`` and ``vq`` survive CASE(1) only in
    the commented-out ``vflx`` line at ``:2047``, and ``cldfra_bl1D`` is read
    only by CASE(2) at ``:2160``: those five are dead because of the branch.
    ``xland`` (declared ``intent(in)`` at ``:1850``/``:1874``) and ``dx``
    (``:1851``/``:1875``) are dead for a stronger reason -- no CASE reads
    either one anywhere in ``mym_length``'s body, so dropping them does not
    depend on ``bl_mynn_mixlength=1``.  ``dtv(kts)`` is likewise never read,
    which is what lets ``mym_initialize`` call this with the level-2 work
    array whose first element mym_level2 leaves untouched.
    """

    nz = dz.size
    qkw = np.empty(nz, dtype=np.float32)
    qtke = np.empty(nz, dtype=np.float32)
    thetaw = np.empty(nz, dtype=np.float32)
    ugrid = F(np.sqrt(F(u[0] * u[0] + v[0] * v[0])))
    wt_u = F(F(1.0) - min(max(F(ugrid - F(15.0)), F(0.0)) / F(30.0),
                          F(0.5)))
    alp3 = F(F(2.5) * wt_u)
    zi2 = max(F(zi), F(300.0))
    h1 = min(max(F(F(0.3) * zi2), F(300.0)), F(600.0))
    h2 = F(h1 / F(2.0))
    qtke[0] = max(F(F(0.5) * qke[0]), F(0.5e-3))
    thetaw[0] = theta[0]
    qkw[0] = F(np.sqrt(max(qke[0], F(1.0e-3))))
    for k in range(1, nz):
        afk = F(dz[k] / F(dz[k] + dz[k - 1]))
        abk = F(F(1.0) - afk)
        qkw[k] = F(np.sqrt(max(
            F(qke[k] * abk + qke[k - 1] * afk), F(1.0e-3)
        )))
        qtke[k] = max(F(F(0.5) * F(qkw[k] * qkw[k])), F(0.005))
        thetaw[k] = F(theta[k] * abk + theta[k - 1] * afk)

    elt = F(1.0e-5)
    vsc_sum = F(1.0e-5)
    k = 1
    # THE MODEL TOP.  WRF's ``DO WHILE (zwk .LE. zi2+h1)`` has no upper
    # bound: when the boundary layer plus its entrainment layer reaches the
    # model top it reads ``dz`` and ``qkw`` one level past the column, which
    # is undefined.  The integral ends at the top interior interface
    # instead, as the CUDA kernel's does (``k < nz``), so both give the
    # whole-column integral there.  This reference used to raise "column
    # top is too low" at that point while the device ran on.
    while k < nz and F(zw[k]) <= F(zi2 + h1):
        dzk = F(F(0.5) * F(dz[k] + dz[k - 1]))
        qdz = F(min(max(qkw[k], F(0.01)), F(30.0)) * dzk)
        elt = F(elt + F(qdz * zw[k]))
        vsc_sum = F(vsc_sum + qdz)
        k += 1
    elt = min(max(F(F(0.23) * elt / vsc_sum), F(8.0)), F(400.0))
    vsc = _powf(F(GTR * elt * max(F(fltv), F(0.0))), ONETHIRD)
    _, elblavg = _boulac_length(zw, dz, qtke, thetaw, boulac_down_sign)
    el = np.empty(nz, dtype=np.float32)
    el[0] = F(0.0)
    for k in range(1, nz):
        zwk = F(zw[k])
        if dtv[k] > F(0.0):
            bv = max(F(np.sqrt(F(GTR * dtv[k]))), F(0.0001))
            numerator = max(
                F(F(0.3) * max(qkw[k], F(0.018))),
                F(F(50.0) * edmf_a[k - 1] * edmf_w[k - 1]),
            )
            elb = F(numerator / bv * F(
                F(1.0) + alp3 * F(np.sqrt(F(vsc / F(bv * elt))))
            ))
            elb = min(elb, zwk)
            elf = F(max(qkw[k], F(0.018)) / bv)
            elblavg[k] = max(
                elblavg[k],
                F(F(50.0) * edmf_a[k - 1] * edmf_w[k - 1] / bv),
            )
        else:
            elb = F(1.0e10)
            elf = elb
        if rmo > F(0.0):
            els = F(F(0.4) * zwk
                    / F(F(1.0) + F(3.5) * min(F(zwk * rmo), F(1.0))))
        else:
            els = F(F(0.4) * zwk
                    * _powf(F(F(1.0) - F(5.0) * zwk * rmo), F(0.2)))
        weight = F(F(0.5) * _tanhf(F(
            (zwk - F(zi2 + h1)) / h2
        )) + F(0.5))
        value = F(np.sqrt(F(els * els / F(
            F(1.0) + F(els * els / F(elt * elt))
        ))))
        value = min(value, elb)
        value = min(value, elf)
        value = F(value * F(F(1.0) - weight)
                  + F(0.3) * elblavg[k] * weight)
        el[k] = F(value * F(psig_bl))
    return el, qkw


def _mym_length_local_column(
    dz, zw, qke, dtv, edmf_w, edmf_a, rmo, fltv, zi, psig_bl,
):
    """WRF v4.6.1 ``mym_length`` CASE(2), lines 2100-2232.

    Transcribed under ``licenses/LICENSE-WRF-public-domain.txt``. The source's
    Ugrid, Uonset, cldavg and elb stable-branch temporaries never feed an
    output. Keep the FP32 operator order of the remaining expressions.
    """
    nz = dz.size
    qkw = np.empty(nz, dtype=np.float32)
    qtke = np.empty(nz, dtype=np.float32)
    zi2 = max(zi, F(300.0))
    h1 = min(max(F(F(0.3) * zi2), F(300.0)), F(600.0))
    h2 = F(h1 * F(0.5))
    qtke[0] = max(F(F(0.5) * qke[0]), F(F(0.5) * QKEMIN))
    qkw[0] = F(np.sqrt(max(qke[0], QKEMIN)))
    for k in range(1, nz):
        afk = F(dz[k] / F(dz[k] + dz[k - 1]))
        abk = F(F(1.0) - afk)
        qkw[k] = F(np.sqrt(max(
            F(F(qke[k] * abk) + F(qke[k - 1] * afk)), QKEMIN)))
        qtke[k] = F(F(0.5) * F(qkw[k] * qkw[k]))
    elt = F(1.0e-5)
    vsc_sum = F(1.0e-5)
    pblh_plus_ent = max(F(zi + h1), F(100.0))
    k = 1
    # The model top, as in CASE(1): WRF's unbounded DO WHILE reads past the
    # column when PBLH plus the entrainment layer reaches the top, and the
    # integral ends at the top interior interface here and on the device.
    while k < nz and zw[k] <= pblh_plus_ent:
        dzk = F(F(0.5) * F(dz[k] + dz[k - 1]))
        qdz = F(min(max(qkw[k], F(0.03)), F(30.0)) * dzk)
        elt = F(elt + F(qdz * zw[k]))
        vsc_sum = F(vsc_sum + qdz)
        k += 1
    elt = min(max(F(F(F(0.22) * elt) / vsc_sum), F(10.0)), F(400.0))
    vsc = _powf(F(F(GTR * elt) * max(fltv, F(0.0))), ONETHIRD)
    el = np.empty(nz, dtype=np.float32)
    el[0] = F(0.0)
    for k in range(1, nz):
        zwk = zw[k]
        dzk = F(F(0.5) * F(dz[k] + dz[k - 1]))
        wstar = F(F(1.25) * _powf(
            F(F(GTR * zi) * max(fltv, F(1.0e-4))), ONETHIRD))
        weight = F(F(F(0.5) * _tanhf(F(
            F(zwk - F(zi2 + h1)) / h2))) + F(0.5))
        if dtv[k] > F(0.0):
            bv = max(F(np.sqrt(F(GTR * dtv[k]))), F(0.001))
            numerator = max(F(F(0.30) * qkw[k]),
                            F(F(F(50.0) * edmf_a[k - 1]) * edmf_w[k - 1]))
            elb_mf = F(F(numerator / bv) * F(F(1.0) + F(F(2.0) * F(
                np.sqrt(F(vsc / F(bv * elt)))))))
            tau = min(max(F(F(F(1000.0) * wstar) / F(9.81)), F(30.0)),
                      F(150.0))
            tau = F(F(tau * F(F(1.0) - weight)) + F(F(50.0) * weight))
            elf = min(max(F(tau * F(np.sqrt(min(qtke[k], F(40.0))))),
                          F(F(F(F(50.0) * edmf_a[k]) * edmf_w[k]) / bv)),
                      zwk)
        else:
            tau = min(max(F(F(F(1000.0) * wstar) / F(9.81)), F(50.0)),
                      F(200.0))
            tau = F(F(tau * F(F(1.0) - weight))
                    + F(max(F(100.0), F(dzk * F(0.25))) * weight))
            elf = min(F(tau * F(np.sqrt(min(qtke[k], F(40.0))))), zwk)
            elb_mf = elf
        elf = F(elf / F(F(1.0) + F(elf / F(800.0))))
        elb_mf = max(elb_mf, F(0.01))
        if rmo > F(0.0):
            els = F(F(KARMAN * zwk) / F(F(1.0) + F(
                F(3.5) * min(F(zwk * rmo), F(1.0)))))
        else:
            els = F(F(KARMAN * zwk) * _powf(
                F(F(1.0) - F(F(F(5.0) * zwk) * rmo)), F(0.2)))
        els2 = F(els * els)
        value = F(np.sqrt(F(els2 / F(
            F(F(1.0) + F(els2 / F(elt * elt)))
            + F(els2 / F(elb_mf * elb_mf))))))
        value = F(F(value * F(F(1.0) - weight)) + F(elf * weight))
        el_les = min(F(els / F(F(1.0) + F(els / F(12.0)))), elb_mf)
        el[k] = F(F(value * psig_bl) + F(F(F(1.0) - psig_bl) * el_les))
    return el, qkw


def mynn_mixlength_default(
    values: Mapping[str, object], *, bl_mynn_mixlength: int = 1,
    boulac_down_sign: bool = False,
) -> dict[str, np.ndarray]:
    """Translate WRF ``mym_length`` options 1 (nonlocal) and 2 (local).

    ``boulac_down_sign`` is correct-mode ML1 (:func:`_boulac_length`).
    """

    if type(bl_mynn_mixlength) is not int or bl_mynn_mixlength not in (1, 2):
        raise ValueError("MYNN mixing length requires bl_mynn_mixlength=1 or 2")

    missing = [name for name in MYNN_MIXLENGTH_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN mixing-length inputs: {', '.join(missing)}")
    column_names = (
        "dz", "u", "v", "qke", "dtv", "theta", "vt", "vq",
        "cldfra", "edmf_w", "edmf_a",
    )
    columns = {
        name: np.asarray(values[name], dtype=np.float32) for name in column_names
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError("MYNN mixing-length columns must share shape (ncol,nz)")
    ncol, nz = next(iter(shapes))
    interface = np.asarray(values["zw"], dtype=np.float32)
    if nz < 3 or interface.shape != (ncol, nz + 1):
        raise ValueError("MYNN mixing-length zw must have shape (ncol,nz+1)")
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi", "psig_bl",
    )
    scalars = {}
    for name in scalar_names:
        try:
            scalars[name] = np.broadcast_to(
                np.asarray(values[name], dtype=np.float32), (ncol,)
            )
        except ValueError as exc:
            raise ValueError(f"{name} is not broadcastable to ncol") from exc
    if any(not array.size or not np.isfinite(array).all()
           for array in (*columns.values(), interface, *scalars.values())):
        raise ValueError("MYNN mixing-length inputs must be finite")
    if np.any(columns["dz"] <= 0.0):
        raise ValueError("MYNN mixing-length layer depths must be positive")

    el_out = np.empty((ncol, nz), dtype=np.float32)
    qkw_out = np.empty((ncol, nz), dtype=np.float32)
    for column in range(ncol):
        if bl_mynn_mixlength == 2:
            el_out[column], qkw_out[column] = _mym_length_local_column(
                columns["dz"][column], interface[column], columns["qke"][column],
                columns["dtv"][column], columns["edmf_w"][column],
                columns["edmf_a"][column], F(scalars["rmo"][column]),
                F(scalars["fltv"][column]), F(scalars["zi"][column]),
                F(scalars["psig_bl"][column]),
            )
            continue
        el, qkw = _mym_length_column(
            columns["dz"][column], interface[column], columns["u"][column],
            columns["v"][column], columns["qke"][column],
            columns["dtv"][column], columns["theta"][column],
            columns["edmf_w"][column], columns["edmf_a"][column],
            F(scalars["rmo"][column]), F(scalars["fltv"][column]),
            F(scalars["zi"][column]), F(scalars["psig_bl"][column]),
            boulac_down_sign,
        )
        el_out[column] = el
        qkw_out[column] = qkw
    return {"el": el_out, "qkw": qkw_out}


def _mym_level2_column(
    dz: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    thl: np.ndarray,
    qw: np.ndarray,
    vt: np.ndarray,
    vq: np.ndarray,
    dtl: np.ndarray,
    dqw: np.ndarray,
    dtv: np.ndarray,
    gm: np.ndarray,
    gh: np.ndarray,
    sm: np.ndarray,
    sh: np.ndarray,
) -> None:
    """Single-column WRF ``mym_level2``, ``module_bl_mynn.F:1766-1820``.

    The Fortran loop runs ``kts+1 .. kte``, so element ``0`` of every output
    is left exactly as the caller supplied it.  ``mym_initialize`` relies on
    that: it hands WRF's ``sh``/``sm`` dummy arguments straight back with
    their incoming surface value intact.
    """

    nz = dz.size
    for k in range(1, nz):
        dzk = F(F(0.5) * F(dz[k] + dz[k - 1]))
        afk = F(dz[k] / F(dz[k] + dz[k - 1]))
        abk = F(F(1.0) - afk)
        du = F(F(u[k]) - F(u[k - 1]))
        dv = F(F(v[k]) - F(v[k - 1]))
        duz = F(F(du * du) + F(dv * dv))
        duz = F(duz / F(dzk * dzk))
        dtz = F(F(F(thl[k]) - F(thl[k - 1])) / dzk)
        dqz = F(F(F(qw[k]) - F(qw[k - 1])) / dzk)
        vtt = F(F(1.0) + F(F(vt[k]) * abk) + F(F(vt[k - 1]) * afk))
        vqq = F(TV0 + F(F(vq[k]) * abk) + F(F(vq[k - 1]) * afk))
        dtq = F(F(vtt * dtz) + F(vqq * dqz))
        level_gh = F(-F(dtq * GTR))
        ri = F(-level_gh / max(duz, F(1.0e-10)))
        a2fac = F(F(1.0) / F(F(1.0) + max(ri, F(0.0))))

        rfc = F(G1 / F(G1 + G2))
        f1 = F(
            B1 * F(G1 - C1)
            + F(3.0) * A2 * a2fac * F(F(1.0) - C2) * F(F(1.0) - C5)
            + F(2.0) * A1 * F(F(3.0) - F(2.0) * C2)
        )
        f2 = F(B1 * F(G1 + G2) - F(3.0) * A1 * F(F(1.0) - C2))
        rf1 = F(B1 * F(G1 - C1) / f1)
        rf2 = F(B1 * G1 / f2)
        smc = F(A1 / F(A2 * a2fac) * f1 / f2)
        shc = F(F(3.0) * F(A2 * a2fac) * F(G1 + G2))
        ri1 = F(F(0.5) / smc)
        ri2 = F(rf1 * smc)
        ri3 = F(F(4.0) * rf2 * smc - F(2.0) * ri2)
        ri4 = F(ri2 * ri2)
        radical = F(ri * ri - ri3 * ri + ri4)
        rf = min(F(ri1 * F(ri + ri2 - F(np.sqrt(radical)))), rfc)

        dtl[k] = dtz
        dqw[k] = dqz
        dtv[k] = dtq
        gm[k] = duz
        gh[k] = level_gh
        sh[k] = F(shc * F(rfc - rf) / F(F(1.0) - rf))
        sm[k] = F(smc * F(rf1 - rf) / F(rf2 - rf) * sh[k])


def mynn_initialize_default(
    values: Mapping[str, object],
    *,
    initialize_qke: bool = True,
    bl_mynn_mixlength: int = 1,
    spp_pbl: int = 0,
    boulac_down_sign: bool = False,
) -> dict[str, np.ndarray]:
    """Translate WRF ``mym_initialize`` (``module_bl_mynn.F:1514-1674``).

    The routine seeds ``qke`` and the mixing-length state on the first step by
    iterating ``mym_length`` five times against the level-2 production terms.
    ``pmz``, ``phh``, ``flt``, ``fltv`` and ``flq`` are local variables with
    initialisers inside the Fortran (``:1545-1546``): they are 1, 1, 0, 0 and 0
    and are never assigned, so the surface-flux seeds of ``tsq``, ``qsq`` and
    ``cov`` collapse to zero and ``mym_length`` always sees ``fltv = 0``.

    ``INITIALIZE_QKE`` selects whether ``qke`` is seeded or read; both paths
    are transcribed.  ``sm``/``sh`` are WRF dummy arguments that ``mym_level2``
    fills only from ``kts+1``, so their surface element is returned unchanged;
    ``el`` is fully rewritten by ``mym_length`` and needs no input.

    ``xland``, ``dx``, ``thetav``, ``cldfra`` and the stochastic column reach
    the Fortran but none of them reaches an output. Four are dead outright,
    whatever ``bl_mynn_mixlength`` selects: ``xland`` (``:1515``/``:1533``)
    and ``dx`` (``:1516``/``:1534``) are only handed on to ``mym_length`` at
    ``:1601-1602``, which declares them ``intent(in)`` at ``:1850-1851``
    /``:1874-1875`` and then reads neither in any CASE; ``thetav``
    (``:1519``/``:1547``) is only handed to ``mym_level2`` at ``:1561``, which
    mentions it only in the commented-out ``:1779`` line; and ``rstoch_col``
    (``:1525``/``:1548``) is never passed on or read at all.  ``cldfra_bl1D``
    (``:1521``/``:1538``, forwarded at ``:1609``) is read at ``:2160`` in
    CASE(2), but only to form the unused ``cldavg`` temporary. It therefore
    reaches no output for either admitted option. They stay in the signature
    so the contract matches WRF's.
    """

    if type(bl_mynn_mixlength) is not int or bl_mynn_mixlength not in (1, 2):
        raise ValueError("MYNN initialize requires bl_mynn_mixlength=1 or 2")
    if type(spp_pbl) is not int or spp_pbl not in (0, 1):
        raise ValueError("MYNN initialize requires spp_pbl in {0,1}")
    if type(initialize_qke) is not bool:
        raise TypeError("initialize_qke must be a bool")
    missing = [name for name in MYNN_INITIALIZE_INPUTS if name not in values]
    if missing:
        raise TypeError(
            f"missing MYNN initialize inputs: {', '.join(missing)}")

    columns = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in MYNN_INITIALIZE_COLUMN_INPUTS
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError("MYNN initialize columns must share shape (ncol,nz)")
    ncol, nz = next(iter(shapes))
    if nz < 4:
        raise ValueError("MYNN initialize requires nz >= 4")
    interface = np.asarray(values["zw"], dtype=np.float32)
    if interface.shape != (ncol, nz + 1):
        raise ValueError("MYNN initialize zw must have shape (ncol,nz+1)")
    scalars: dict[str, np.ndarray] = {}
    for name in MYNN_INITIALIZE_SCALAR_INPUTS:
        try:
            scalars[name] = np.broadcast_to(
                np.asarray(values[name], dtype=np.float32), (ncol,)
            )
        except ValueError as exc:
            raise ValueError(f"{name} is not broadcastable to ncol") from exc
    if any(not np.isfinite(array).all()
           for array in (*columns.values(), interface, *scalars.values())):
        raise ValueError("MYNN initialize inputs must be finite")
    if np.any(columns["dz"] <= 0.0):
        raise ValueError("MYNN initialize layer depths must be positive")
    if np.any(scalars["ust"] <= 0.0):
        raise ValueError("MYNN initialize requires a positive ust")

    outputs = {
        name: np.empty((ncol, nz), dtype=np.float32)
        for name in MYNN_INITIALIZE_OUTPUTS
    }
    for column in range(ncol):
        dz = columns["dz"][column]
        zw = interface[column]
        u = columns["u"][column]
        v = columns["v"][column]
        thl = columns["thl"][column]
        qw = columns["qw"][column]
        theta = columns["theta"][column]
        edmf_w = columns["edmf_w"][column]
        edmf_a = columns["edmf_a"][column]
        ust = F(scalars["ust"][column])
        rmo = F(scalars["rmo"][column])
        zi = F(scalars["zi"][column])
        psig_bl = F(scalars["psig_bl"][column])

        # module_bl_mynn.F:1552-1556 zeroes ql, vt and vq before level 2.
        vt = np.zeros(nz, dtype=np.float32)
        vq = np.zeros(nz, dtype=np.float32)
        dtl = np.zeros(nz, dtype=np.float32)
        dqw = np.zeros(nz, dtype=np.float32)
        dtv = np.zeros(nz, dtype=np.float32)
        gm = np.zeros(nz, dtype=np.float32)
        gh = np.zeros(nz, dtype=np.float32)
        sm = columns["sm"][column].astype(np.float32).copy()
        sh = columns["sh"][column].astype(np.float32).copy()
        _mym_level2_column(dz, u, v, thl, qw, vt, vq,
                           dtl, dqw, dtv, gm, gh, sm, sh)

        qke = columns["qke"][column].astype(np.float32).copy()
        el = np.zeros(nz, dtype=np.float32)
        tsq = np.zeros(nz, dtype=np.float32)
        qsq = np.zeros(nz, dtype=np.float32)
        cov = np.zeros(nz, dtype=np.float32)
        if initialize_qke:
            qke[0] = F(F(F(1.5) * F(ust * ust)) * _powf(B1_PMZ, TWOTHIRDS))
            for k in range(1, nz):
                taper = F(F(F(ust * F(700.0)) - F(zw[k]))
                          / F(max(ust, F(0.01)) * F(700.0)))
                qke[k] = F(qke[0] * max(taper, F(0.01)))
        # phm and the flt/flq seeds of tsq, qsq and cov: flt = flq = 0.
        phm = F(PHH_B2 / _powf(B1_PMZ, ONETHIRD))
        tsq[0] = F(phm * F(F(F(0.0) / ust) * F(F(0.0) / ust)))
        qsq[0] = F(phm * F(F(F(0.0) / ust) * F(F(0.0) / ust)))
        cov[0] = F(F(phm * F(F(0.0) / ust)) * F(F(0.0) / ust))
        for k in range(1, nz):
            vkz = F(KARMAN * F(zw[k]))
            el[k] = F(vkz / F(F(1.0) + F(vkz / F(100.0))))

        pdk = np.zeros(nz, dtype=np.float32)
        pdt = np.zeros(nz, dtype=np.float32)
        pdq = np.zeros(nz, dtype=np.float32)
        pdc = np.zeros(nz, dtype=np.float32)
        for _ in range(MYM_INITIALIZE_ITERATIONS):
            if bl_mynn_mixlength == 2:
                el, qkw = _mym_length_local_column(
                    dz, zw, qke, dtv, edmf_w, edmf_a,
                    rmo, F(0.0), zi, psig_bl,
                )
            else:
                el, qkw = _mym_length_column(
                    dz, zw, u, v, qke, dtv, theta, edmf_w, edmf_a,
                    rmo, F(0.0), zi, psig_bl, boulac_down_sign,
                )
            for k in range(1, nz):
                elq = F(el[k] * qkw[k])
                pdk[k] = F(elq * F(F(sm[k] * gm[k]) + F(sh[k] * gh[k])))
                pdt[k] = F(F(elq * sh[k]) * F(dtl[k] * dtl[k]))
                pdq[k] = F(F(elq * sh[k]) * F(dqw[k] * dqw[k]))
                pdc[k] = F(F(F(elq * sh[k]) * dtl[k]) * dqw[k])

            vkz = F(F(KARMAN * F(0.5)) * F(dz[0]))
            elv = F(F(F(0.5) * F(el[1] + el[0])) / vkz)
            if initialize_qke:
                floor = max(ust, F(0.02))
                qke[0] = F(
                    F(F(1.0) * F(floor * floor))
                    * _powf(F(B1_PMZ * elv), TWOTHIRDS)
                )
            phm = F(PHH_B2 / _powf(F(B1_PMZ / F(elv * elv)), ONETHIRD))
            tsq[0] = F(phm * F(F(F(0.0) / ust) * F(F(0.0) / ust)))
            qsq[0] = F(phm * F(F(F(0.0) / ust) * F(F(0.0) / ust)))
            cov[0] = F(F(phm * F(F(0.0) / ust)) * F(F(0.0) / ust))

            for k in range(1, nz - 1):
                b1l = F(F(B1 * F(0.25)) * F(el[k + 1] + el[k]))
                tmpq = min(max(F(b1l * F(pdk[k + 1] + pdk[k])), QKEMIN),
                           F(125.0))
                if initialize_qke:
                    qke[k] = _powf(tmpq, TWOTHIRDS)
                if qke[k] <= F(0.0):
                    b2l = F(0.0)
                else:
                    b2l = F(F(B2 * F(b1l / B1)) / F(np.sqrt(qke[k])))
                tsq[k] = F(b2l * F(pdt[k + 1] + pdt[k]))
                qsq[k] = F(b2l * F(pdq[k + 1] + pdq[k]))
                cov[k] = F(b2l * F(pdc[k + 1] + pdc[k]))

        if initialize_qke:
            qke[0] = F(F(0.5) * F(qke[0] + qke[1]))
            qke[nz - 1] = qke[nz - 2]
        tsq[nz - 1] = tsq[nz - 2]
        qsq[nz - 1] = qsq[nz - 2]
        cov[nz - 1] = cov[nz - 2]

        outputs["el"][column] = el
        outputs["qke"][column] = qke
        outputs["tsq"][column] = tsq
        outputs["qsq"][column] = qsq
        outputs["cov"][column] = cov
        outputs["sm"][column] = sm
        outputs["sh"][column] = sh
    return outputs


def mynn_turbulence_default(
    values: Mapping[str, object],
    *,
    closure: float = 2.6,
    bl_mynn_mixlength: int = 1,
    spp_pbl: int = 0,
) -> dict[str, np.ndarray]:
    """Translate default WRF ``mym_turbulence`` for complete columns.

    The admitted identity fixes ``closure=2.6``, mixing-length options 1 or
    2, TKE-budget output off, and stochastic PBL perturbations off unless
    spp_pbl=1 supplies rstoch. EDMF and
    cloud fractions still enter the source-defined diffusivity floor; they
    are therefore required inputs rather than silently set to zero here.
    """

    from woof.core.spp_kernel_sources import spp_flag
    stochastic = spp_flag(spp_pbl, "spp_pbl")
    missing = [name for name in MYNN_TURBULENCE_INPUTS if name not in values]
    if missing:
        raise TypeError(
            f"missing MYNN turbulence inputs: {', '.join(missing)}"
        )
    if not np.isfinite(closure) or float(closure) != 2.6:
        raise ValueError("MYNN first turbulence lane requires closure=2.6")
    column_names = (
        "dz", "u", "v", "thl", "thetav", "ql", "qw", "qke", "tsq",
        "qsq", "cov", "vt", "vq", "theta", "cldfra", "edmf_w",
        "edmf_a", "tkeprodtd",
    )
    columns = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in column_names
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError("MYNN turbulence columns must share shape (ncol,nz)")
    ncol, nz = next(iter(shapes))
    zw = np.asarray(values["zw"], dtype=np.float32)
    if nz < 3 or zw.shape != (ncol, nz + 1):
        raise ValueError("MYNN turbulence zw must have shape (ncol,nz+1)")
    rstoch = None
    if stochastic:
        if "rstoch" not in values:
            raise TypeError("MYNN stochastic turbulence requires rstoch[ncol,nz]")
        rstoch = np.asarray(values["rstoch"], dtype=np.float32)
        if rstoch.shape != (ncol, nz) or not np.isfinite(rstoch).all():
            raise ValueError("MYNN stochastic turbulence rstoch must be finite with shape (ncol,nz)")
    scalar_names = (
        "xland", "dx", "rmo", "flt", "fltv", "flq", "zi",
        "psig_bl", "psig_shcu",
    )
    scalars: dict[str, np.ndarray] = {}
    for name in scalar_names:
        try:
            scalars[name] = np.broadcast_to(
                np.asarray(values[name], dtype=np.float32), (ncol,)
            )
        except ValueError as exc:
            raise ValueError(f"{name} is not broadcastable to ncol") from exc
    if any(not np.isfinite(array).all()
           for array in (*columns.values(), zw, *scalars.values())):
        raise ValueError("MYNN turbulence inputs must be finite")
    if np.any(columns["dz"] <= 0.0):
        raise ValueError("MYNN turbulence layer depths must be positive")
    if np.any(scalars["dx"] <= 0.0) or np.any(scalars["zi"] <= 0.0):
        raise ValueError("MYNN turbulence dx and zi must be positive")

    # mym_level2 defines interfaces kts+1:kte.  Flattening all adjacent
    # pairs keeps the scalar FP32 operation order of the already-oracled
    # primitive while evaluating every independent column at once.
    pair_values: dict[str, np.ndarray] = {}
    for name in ("dz", "u", "v", "thl", "thetav", "qw", "ql", "vt", "vq"):
        pair_values[name] = columns[name][:, 1:].reshape(-1)
        pair_values[f"{name}_prev"] = columns[name][:, :-1].reshape(-1)
    level2_flat = mynn_level2_pairs(pair_values)
    level2 = {
        name: np.zeros((ncol, nz), dtype=np.float32)
        for name in MYNN_LEVEL2_OUTPUTS
    }
    for name in MYNN_LEVEL2_OUTPUTS:
        level2[name][:, 1:] = level2_flat[name].reshape(ncol, nz - 1)

    length = mynn_mixlength_default({
        "dz": columns["dz"], "zw": zw, "u": columns["u"],
        "v": columns["v"], "qke": columns["qke"],
        "dtv": level2["dtv"], "theta": columns["theta"],
        "vt": columns["vt"], "vq": columns["vq"],
        "cldfra": columns["cldfra"], "edmf_w": columns["edmf_w"],
        "edmf_a": columns["edmf_a"],
        **scalars,
    }, bl_mynn_mixlength=bl_mynn_mixlength)
    outputs = {
        name: np.zeros((ncol, nz), dtype=np.float32)
        for name in (
            "dfm", "dfh", "dfq", "pdk", "pdt", "pdq", "pdc", "tcd",
            "qcd", "el", "qkw", "sm", "sh", "dtl", "dqw", "dtv",
            "gm", "gh",
        )
    }
    outputs["el"][...] = length["el"]
    outputs["qkw"][...] = length["qkw"]
    for name in MYNN_LEVEL2_OUTPUTS:
        outputs[name][...] = level2[name]

    for column in range(ncol):
        dz = columns["dz"][column]
        u = columns["u"][column]
        v = columns["v"][column]
        el = outputs["el"][column]
        qkw = outputs["qkw"][column]
        sm = outputs["sm"][column]
        sh = outputs["sh"][column]
        gm = outputs["gm"][column]
        gh = outputs["gh"][column]
        dtl = outputs["dtl"][column]
        dqw = outputs["dqw"][column]
        cldfra = columns["cldfra"][column]
        edmf_w = columns["edmf_w"][column]
        edmf_a = columns["edmf_a"][column]
        tkeprodtd = columns["tkeprodtd"][column]
        for k in range(1, nz):
            dzk = F(F(0.5) * F(dz[k] + dz[k - 1]))
            elsq = np.float64(F(el[k] * el[k]))
            q3sq = np.float64(F(qkw[k] * qkw[k]))
            source = F(F(sm[k] * gm[k]) + F(sh[k] * gh[k]))
            q2sq = np.float64(B1) * elsq * np.float64(source)

            # module_bl_mynn.F:2734 floors sh before the branch, after q2sq
            # has already been formed from the unfloored value.  It survives
            # only down the Helfand-Labraga path, where sh is scaled rather
            # than recomputed, but there it is the difference between a
            # vanishing dfh and an exactly zero one.
            sh[k] = max(sh[k], F(1.0e-5))

            # Canuto/Kitamura modification and Helfand-Labraga limiter.
            du = F(u[k] - u[k - 1])
            dv = F(v[k] - v[k - 1])
            duz = F(F(du * du) + F(dv * dv))
            duz = F(duz / F(dzk * dzk))
            ri = F(-F(gh[k]) / max(duz, F(1.0e-10)))
            a2fac = F(F(1.0) / F(F(1.0) + max(ri, F(0.0))))
            # ``a2fac**2`` is a real(kind_phys) square in the Fortran, so it
            # rounds to FP32 before the DOUBLE PRECISION product widens it.
            # Squaring in FP64 is exact and therefore a different value.
            a2fac_sq = np.float64(F(a2fac * a2fac))
            gmel = np.float64(gm[k]) * elsq
            ghel = np.float64(gh[k]) * elsq
            if q3sq / elsq < -np.float64(gh[k]):
                q3sq = -elsq * np.float64(gh[k])

            if q3sq < q2sq:
                qdiv = np.sqrt(q3sq / q2sq)
                sh[k] = F(np.float64(sh[k]) * qdiv)
                sm[k] = F(np.float64(sm[k]) * qdiv)
                e1 = q3sq - np.float64(E1C) * ghel * np.float64(a2fac) * qdiv**2
                e2 = q3sq - np.float64(E2C) * ghel * np.float64(a2fac) * qdiv**2
                e3 = e1 + np.float64(E3C) * ghel * a2fac_sq * qdiv**2
                e4 = e1 - np.float64(E4C) * ghel * np.float64(a2fac) * qdiv**2
                eden = e2 * e4 + e3 * np.float64(E5C) * gmel * qdiv**2
                if eden < 1.0e-20:
                    eden = 1.0e-20
            else:
                qdiv = 1.0
                e1 = q3sq - np.float64(E1C) * ghel * np.float64(a2fac)
                e2 = q3sq - np.float64(E2C) * ghel * np.float64(a2fac)
                e3 = e1 + np.float64(E3C) * ghel * a2fac_sq
                e4 = e1 - np.float64(E4C) * ghel * np.float64(a2fac)
                eden = e2 * e4 + e3 * np.float64(E5C) * gmel
                if eden < 1.0e-20:
                    eden = 1.0e-20
                sm[k] = F(
                    q3sq * np.float64(A1)
                    * (e3 - np.float64(THREE_C1) * e4) / eden
                )
                sh[k] = F(
                    q3sq * np.float64(F(A2 * a2fac))
                    * (e2 + np.float64(THREE_C1_E5C) * gmel)
                    / eden
                )

            sh[k] = min(max(sh[k], F(0.0)), F(4.0))
            sm[k] = min(sm[k], F(5.0) * max(sh[k], F(0.02)))
            cldavg = F(F(0.5) * F(cldfra[k - 1] + cldfra[k]))
            if edmf_a[k] > F(0.001) or cldavg > F(0.02):
                plume_floor = F(F(0.03) * min(
                    F(F(10.0) * edmf_a[k] * edmf_w[k]), F(1.0)
                ))
                cloud_floor = F(F(0.05) * min(cldavg, F(1.0)))
                sm[k] = max(sm[k], plume_floor, cloud_floor)
                sh[k] = max(sh[k], plume_floor, cloud_floor)

            elq = F(el[k] * qkw[k])
            elh = F(np.float64(elq) * qdiv)
            outputs["pdk"][column, k] = F(
                F(elq * F(F(sm[k] * gm[k]) + F(sh[k] * gh[k])))
                + F(F(0.5) * tkeprodtd[k])
            )
            outputs["pdt"][column, k] = F(
                elh * F(sh[k] * dtl[k]) * dtl[k]
            )
            outputs["pdq"][column, k] = F(
                elh * F(sh[k] * dqw[k]) * dqw[k]
            )
            outputs["pdc"][column, k] = F(
                F(elh * F(sh[k] * dtl[k]) * dqw[k] * F(0.5))
                + F(elh * F(sh[k] * dqw[k]) * dtl[k] * F(0.5))
            )
            # Counter-gradient terms are zero for the admitted level-2.6
            # closure; retain explicit arrays because mynn_tendencies takes
            # them as part of the full WRF ABI.
            outputs["tcd"][column, k] = F(0.0)
            outputs["qcd"][column, k] = F(0.0)
            outputs["dfm"][column, k] = F(elq * sm[k] / dzk)
            outputs["dfh"][column, k] = F(elq * sh[k] / dzk)
            outputs["dfq"][column, k] = outputs["dfm"][column, k]

        # WRF differences the counter-gradient flux after imposing zero
        # values at both endpoints.  It is numerically zero for closure 2.6,
        # but the loop is preserved for admission of closure 3 later.
        outputs["tcd"][column, -1] = F(0.0)
        outputs["qcd"][column, -1] = F(0.0)
        for k in range(nz - 1):
            outputs["tcd"][column, k] = F(
                F(outputs["tcd"][column, k + 1]
                  - outputs["tcd"][column, k]) / dz[k]
            )
            outputs["qcd"][column, k] = F(
                F(outputs["qcd"][column, k + 1]
                  - outputs["qcd"][column, k]) / dz[k]
            )
        if stochastic:
            # module_bl_mynn.F:3135-3140. DFQ retains its earlier value.
            for k in range(nz):
                exponent = F(-max(F(zw[column, k] - F(8000.0)), F(0.0)) / F(2000.0))
                taper = max(_expf(exponent), F(0.001))
                for name in ("dfm", "dfh"):
                    value = outputs[name][column, k]
                    outputs[name][column, k] = F(value + F(F(F(value * rstoch[column, k]) * F(1.5)) * taper))
    return outputs


def _tridiag2_fp32(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Translate WRF MYNN ``tridiag2`` without changing FP32 order."""

    n = a.size
    cp = np.empty(n, dtype=np.float32)
    dp = np.empty(n, dtype=np.float32)
    x = np.empty(n, dtype=np.float32)
    cp[0] = F(c[0] / b[0])
    dp[0] = F(d[0] / b[0])
    for k in range(1, n):
        m = F(b[k] - F(cp[k - 1] * a[k]))
        cp[k] = F(c[k] / m)
        dp[k] = F(F(d[k] - F(dp[k - 1] * a[k])) / m)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = F(dp[k] - F(cp[k] * x[k + 1]))
    return x


def mynn_predict_default(
    values: Mapping[str, object],
    *,
    closure: float = 2.6,
    bl_mynn_edmf_tke: int = 0,
    tke_budget: int = 0,
) -> dict[str, np.ndarray]:
    """Translate default WRF ``mym_predict`` prognostic turbulence solve."""

    missing = [name for name in MYNN_PREDICT_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN predictor inputs: {', '.join(missing)}")
    if not np.isfinite(closure) or float(closure) != 2.6:
        raise ValueError("MYNN first predictor lane requires closure=2.6")
    if bl_mynn_edmf_tke != 0 or type(bl_mynn_edmf_tke) is not int:
        raise ValueError("MYNN first predictor lane requires bl_mynn_edmf_tke=0")
    if tke_budget != 0 or type(tke_budget) is not int:
        raise ValueError("MYNN first predictor lane requires tke_budget=0")
    column_names = (
        "dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el", "qke",
        "tsq", "qsq", "cov",
    )
    columns = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in column_names
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError("MYNN predictor columns must share shape (ncol,nz)")
    ncol, nz = next(iter(shapes))
    if nz < 3:
        raise ValueError("MYNN predictor requires nz >= 3")
    interfaces = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in ("s_aw", "s_awqke")
    }
    if any(array.shape != (ncol, nz + 1) for array in interfaces.values()):
        raise ValueError(
            "MYNN predictor s_aw and s_awqke must have shape (ncol,nz+1)"
        )
    scalar_names = ("ust", "flt", "flq", "pmz", "phh", "delt")
    scalars: dict[str, np.ndarray] = {}
    for name in scalar_names:
        try:
            scalars[name] = np.broadcast_to(
                np.asarray(values[name], dtype=np.float32), (ncol,)
            )
        except ValueError as exc:
            raise ValueError(f"{name} is not broadcastable to ncol") from exc
    if any(not np.isfinite(array).all()
           for array in (*columns.values(), *interfaces.values(), *scalars.values())):
        raise ValueError("MYNN predictor inputs must be finite")
    if np.any(columns["dz"] <= 0.0) or np.any(columns["rho"] <= 0.0):
        raise ValueError("MYNN predictor dz and rho must be positive")
    if np.any(scalars["ust"] <= 0.0) or np.any(scalars["delt"] <= 0.0):
        raise ValueError("MYNN predictor ust and delt must be positive")

    qke_out = np.empty((ncol, nz), dtype=np.float32)
    tsq_out = np.empty((ncol, nz), dtype=np.float32)
    qsq_out = np.empty((ncol, nz), dtype=np.float32)
    cov_out = np.empty((ncol, nz), dtype=np.float32)
    for column in range(ncol):
        dz = columns["dz"][column]
        rho = columns["rho"][column]
        dfq = columns["dfq"][column]
        el = columns["el"][column]
        qke = columns["qke"][column].copy()
        tsq = columns["tsq"][column].copy()
        qsq = columns["qsq"][column].copy()
        cov = columns["cov"][column].copy()
        pdk = columns["pdk"][column].copy()
        pdt = columns["pdt"][column].copy()
        pdq = columns["pdq"][column].copy()
        pdc = columns["pdc"][column].copy()
        s_aw = interfaces["s_aw"][column]
        s_awqke = interfaces["s_awqke"][column]
        delt = F(scalars["delt"][column])
        onoff = F(0.0)

        qkw = np.sqrt(np.maximum(qke, F(0.0))).astype(np.float32)
        df3q = np.asarray(F(3.0) * dfq, dtype=np.float32)
        dtz = np.asarray(delt / dz, dtype=np.float32)
        rhoinv = np.empty(nz, dtype=np.float32)
        rhoz = np.empty(nz + 1, dtype=np.float32)
        kqdz = np.empty(nz + 1, dtype=np.float32)
        kmdz = np.empty(nz + 1, dtype=np.float32)
        rhoz[0] = rho[0]
        rhoinv[0] = F(F(1.0) / rho[0])
        kqdz[0] = F(rhoz[0] * df3q[0])
        kmdz[0] = F(rhoz[0] * dfq[0])
        for k in range(1, nz):
            numerator = F(F(rho[k] * dz[k - 1]) + F(rho[k - 1] * dz[k]))
            rhoz[k] = F(numerator / F(dz[k - 1] + dz[k]))
            rhoz[k] = max(rhoz[k], F(1.0e-4))
            rhoinv[k] = F(F(1.0) / max(rho[k], F(1.0e-4)))
            kqdz[k] = F(rhoz[k] * df3q[k])
            kmdz[k] = F(rhoz[k] * dfq[k])
        rhoz[nz] = rhoz[nz - 1]
        kqdz[nz] = F(rhoz[nz] * df3q[nz - 1])
        kmdz[nz] = F(rhoz[nz] * dfq[nz - 1])
        for k in range(1, nz - 1):
            kqdz[k] = max(kqdz[k], F(F(0.5) * s_aw[k]))
            kqdz[k] = max(
                kqdz[k], F(-F(0.5) * F(s_aw[k] - s_aw[k + 1]))
            )
            kmdz[k] = max(kmdz[k], F(F(0.5) * s_aw[k]))
            kmdz[k] = max(
                kmdz[k], F(-F(0.5) * F(s_aw[k] - s_aw[k + 1]))
            )

        ust = F(scalars["ust"][column])
        vkz = F(F(0.4) * F(F(0.5) * dz[0]))
        # gfortran expands the integer power ust**3 as (ust*ust)*ust, two
        # rounded multiplies; NumPy's float32 ** 3 is one powf call, and on
        # the cold-pool family (ust = 0.08) the two land a ULP apart on qke.
        pdk1 = F(F(F(2.0) * F(F(ust * ust) * ust)) * scalars["pmz"][column] / vkz)
        phm = F(F(F(2.0) / ust) * scalars["phh"][column] / vkz)
        pdt1 = F(F(phm * scalars["flt"][column]) * scalars["flt"][column])
        pdq1 = F(F(phm * scalars["flq"][column]) * scalars["flq"][column])
        pdc1 = F(F(phm * scalars["flt"][column]) * scalars["flq"][column])
        del pdt1, pdq1, pdc1
        pdk[0] = F(pdk1 - pdk[1])
        pdt[0] = pdt[1]
        pdq[0] = pdq[1]
        pdc[0] = pdc[1]

        a = np.empty(nz, dtype=np.float32)
        b = np.empty(nz, dtype=np.float32)
        c = np.empty(nz, dtype=np.float32)
        d = np.empty(nz, dtype=np.float32)
        bp = np.empty(nz, dtype=np.float32)
        rp = np.empty(nz, dtype=np.float32)
        for k in range(nz - 1):
            b1l = F(F(24.0) * F(F(0.5) * F(el[k + 1] + el[k])))
            bp[k] = F(F(F(2.0) * qkw[k]) / b1l)
            rp[k] = F(pdk[k + 1] + pdk[k])
            a[k] = F(
                -F(F(dtz[k] * kqdz[k]) * rhoinv[k])
                + F(F(F(F(0.5) * dtz[k]) * rhoinv[k]) * s_aw[k] * onoff)
            )
            b[k] = F(
                F(1.0)
                + F(F(dtz[k] * F(kqdz[k] + kqdz[k + 1])) * rhoinv[k])
                + F(F(F(F(0.5) * dtz[k]) * rhoinv[k])
                    * F(s_aw[k] - s_aw[k + 1]) * onoff)
                + F(F(bp[k] * delt))
            )
            c[k] = F(
                -F(F(dtz[k] * kqdz[k + 1]) * rhoinv[k])
                - F(F(F(F(0.5) * dtz[k]) * rhoinv[k])
                    * s_aw[k + 1] * onoff)
            )
            d[k] = F(
                F(rp[k] * delt) + qke[k]
                + F(F(dtz[k] * rhoinv[k])
                    * F(s_awqke[k] - s_awqke[k + 1]) * onoff)
            )
        a[-1] = F(0.0)
        b[-1] = F(1.0)
        c[-1] = F(0.0)
        d[-1] = qke[-1]
        solved = _tridiag2_fp32(a, b, c, d)
        qke[:] = np.minimum(np.maximum(solved, F(1.0e-3)), F(150.0))

        for k in range(nz - 1):
            b2l = F(F(15.0) * F(F(0.5) * F(el[k + 1] + el[k])))
            bp[k] = F(F(F(2.0) * qkw[k]) / b2l)
            rp[k] = F(pdq[k + 1] + pdq[k])
            a[k] = F(-F(F(dtz[k] * kmdz[k]) * rhoinv[k]))
            b[k] = F(
                F(1.0)
                + F(F(dtz[k] * F(kmdz[k] + kmdz[k + 1])) * rhoinv[k])
                + F(bp[k] * delt)
            )
            c[k] = F(-F(F(dtz[k] * kmdz[k + 1]) * rhoinv[k]))
            d[k] = F(F(rp[k] * delt) + qsq[k])
        a[-1] = F(-1.0)
        b[-1] = F(1.0)
        c[-1] = F(0.0)
        d[-1] = F(0.0)
        qsq[:] = np.maximum(_tridiag2_fp32(a, b, c, d), F(1.0e-17))

        for k in range(nz - 1):
            if qkw[k] <= F(0.0):
                b2l = F(0.0)
            else:
                b2l = F(
                    F(F(15.0) * F(0.25)) * F(el[k + 1] + el[k]) / qkw[k]
                )
            tsq[k] = F(b2l * F(pdt[k + 1] + pdt[k]))
            cov[k] = F(b2l * F(pdc[k + 1] + pdc[k]))
        tsq[-1] = tsq[-2]
        cov[-1] = cov[-2]
        qke_out[column] = qke
        tsq_out[column] = tsq
        qsq_out[column] = qsq
        cov_out[column] = cov
    return {"qke": qke_out, "tsq": tsq_out, "qsq": qsq_out, "cov": cov_out}


def _tanhf(x: np.float32) -> np.float32:
    """Transcription of glibc's FP32 ``tanhf``; see :func:`_expm1f`."""

    x = F(x)
    word = int(np.float32(x).view(np.uint32))
    magnitude = word & 0x7FFFFFFF
    if magnitude < 0x41B00000:  # |x| < 22
        if magnitude < 0x24000000:  # |x| < 2**-55
            return F(x * F(F(1.0) + x))
        if magnitude >= 0x3F800000:  # |x| >= 1
            t = _expm1f(F(F(2.0) * F(abs(x))))
            z = F(F(1.0) - F(F(2.0) / F(t + F(2.0))))
        else:
            t = _expm1f(F(F(-2.0) * F(abs(x))))
            z = F(-t / F(t + F(2.0)))
    else:
        z = F(F(1.0) - EXPM1F_TINY)
    return z if word & 0x80000000 == 0 else F(-z)


def _blend_polynomial(coefficients: tuple, xc: np.float32) -> np.float32:
    """Evaluate the WRF saturation polynomial in the source's Horner order."""

    value = coefficients[8]
    for coefficient in reversed(coefficients[:8]):
        value = F(coefficient + F(xc * value))
    return value


def mynn_esat_blend(t: np.float32) -> np.float32:
    """Translate ``module_bl_mynn.F:esat_blend`` (phase-blended vapour Pa)."""

    t = F(t)
    xc = max(F(-80.0), F(t - T0C))
    if t >= T0C_M6:
        return _blend_polynomial(_ESAT_LIQUID, xc)
    if t <= TICE:
        return _blend_polynomial(_ESAT_ICE, xc)
    esl = _blend_polynomial(_ESAT_LIQUID, xc)
    esi = _blend_polynomial(_ESAT_ICE, xc)
    chi = F(F(T0C_M6 - t) / F(T0C_M6 - TICE))
    return F(F(F(F(1.0) - chi) * esl) + F(chi * esi))


def mynn_qsat_blend(t: np.float32, p: np.float32) -> np.float32:
    """Translate ``module_bl_mynn.F:qsat_blend`` (phase-blended kg/kg)."""

    t = F(t)
    p = F(p)
    xc = max(F(-80.0), F(t - T0C))
    ceiling = F(p * F(0.15))
    if t >= T0C_M6:
        esl = min(_blend_polynomial(_ESAT_LIQUID, xc), ceiling)
        return F(F(F(0.622) * esl) / max(F(p - esl), F(1.0e-5)))
    if t <= TICE:
        esi = min(_blend_polynomial(_ESAT_ICE, xc), ceiling)
        return F(F(F(0.622) * esi) / max(F(p - esi), F(1.0e-5)))
    esl = min(_blend_polynomial(_ESAT_LIQUID, xc), ceiling)
    esi = min(_blend_polynomial(_ESAT_ICE, xc), ceiling)
    rslf = F(F(F(0.622) * esl) / max(F(p - esl), F(1.0e-5)))
    rsif = F(F(F(0.622) * esi) / max(F(p - esi), F(1.0e-5)))
    chi = F(F(T0C_M6 - t) / F(T0C_M6 - TICE))
    return F(F(F(F(1.0) - chi) * rslf) + F(chi * rsif))


def mynn_xl_blend(t: np.float32) -> np.float32:
    """Translate ``module_bl_mynn.F:xl_blend`` (phase-blended latent heat)."""

    t = F(t)
    if t >= T0C:
        return F(XLV + F(CPV_CLIQ * F(t - T0C)))
    if t <= TICE:
        return F(XLS + F(CPV_CICE * F(t - T0C)))
    xlvt = F(XLV + F(CPV_CLIQ * F(t - T0C)))
    xlst = F(XLS + F(CPV_CICE * F(t - T0C)))
    chi = F(F(T0C - t) / F(T0C - TICE))
    return F(F(F(F(1.0) - chi) * xlvt) + F(chi * xlst))


def _condensation_tropopause(th: np.ndarray, p: np.ndarray, nz: int) -> int:
    """Reproduce the ``mym_condensation`` GOTO tropopause search exactly.

    The Fortran loop counts down from ``kte-3`` to ``kts`` and jumps out of the
    loop on the first satisfied level.  When it falls through, the DO variable
    is one below ``kts``, so ``k_tropo`` collapses onto ``kts+2``.  Both exits
    are reproduced here in WRF's one-based level numbering.
    """

    found = 0
    for level in range(nz - 3, 0, -1):
        theta1 = F(th[level - 1])
        theta2 = F(th[level + 1])
        ht1 = F(F(44307.692) * F(F(1.0) - F(
            _powf(F(F(p[level - 1]) / F(101325.0)), F(0.190))
        )))
        ht2 = F(F(44307.692) * F(F(1.0) - F(
            _powf(F(F(p[level + 1]) / F(101325.0)), F(0.190))
        )))
        slope = F(F(theta2 - theta1) / F(ht2 - ht1))
        if slope < TROPO_LAPSE and ht1 < F(19000.0) and ht1 > F(4000.0):
            found = level
            break
    return max(3, found + 2)


def mynn_condensation_default(
    values: Mapping[str, object],
    *,
    bl_mynn_cloudpdf: int = 2,
    spp_pbl: int = 0,
) -> dict[str, np.ndarray]:
    """Translate WRF ``mym_condensation`` for ``bl_mynn_cloudpdf=2``.

    Only the Chaboureau-Bechtold statistical cloud PDF (the WRF Registry
    default) is admitted; ``bl_mynn_cloudpdf`` values 0, 1, and the negative
    mass-flux-isolation test settings are rejected rather than silently
    aliased onto this branch.  Stochastic PBL perturbations are off, so
    ``rstoch`` perturbs the saturation deficit when ``spp_pbl=1``. ``qv``,
    ``sh``, ``el``, ``zw``, ``dx``, ``hfx``, and ``rmo`` are part of WRF's
    argument list but are not read by this branch; they are required so the
    call site matches the Fortran ABI.  ``vt`` and ``vq`` are fully
    overwritten, while ``sgm`` keeps its incoming top-level value because the
    Fortran loop stops at ``kte-1`` and the copy-down never touches it.
    """

    missing = [name for name in MYNN_CONDENSATION_INPUTS if name not in values]
    if missing:
        raise TypeError(
            f"missing MYNN condensation inputs: {', '.join(missing)}"
        )
    if bl_mynn_cloudpdf != 2 or type(bl_mynn_cloudpdf) is not int:
        raise ValueError(
            "MYNN first condensation lane requires bl_mynn_cloudpdf=2"
        )
    if type(spp_pbl) is not int or spp_pbl not in (0, 1):
        raise ValueError("MYNN condensation requires spp_pbl in {0,1}")
    column_names = (
        "dz", "th", "thl", "qw", "qv", "qc", "qi", "qs", "p", "exner",
        "tsq", "qsq", "cov", "sh", "el", "rstoch", "vt", "vq", "sgm",
    )
    columns = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in column_names
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError(
            "MYNN condensation columns must share shape (ncol,nz)"
        )
    ncol, nz = next(iter(shapes))
    if nz < 4:
        raise ValueError("MYNN condensation requires nz >= 4")
    interface = np.asarray(values["zw"], dtype=np.float32)
    if interface.shape != (ncol, nz + 1):
        raise ValueError("MYNN condensation zw must have shape (ncol,nz+1)")
    scalar_names = ("xland", "dx", "pblh", "hfx", "rmo")
    scalars: dict[str, np.ndarray] = {}
    for name in scalar_names:
        try:
            scalars[name] = np.broadcast_to(
                np.asarray(values[name], dtype=np.float32), (ncol,)
            )
        except ValueError as exc:
            raise ValueError(f"{name} is not broadcastable to ncol") from exc
    if any(not np.isfinite(array).all()
           for array in (*columns.values(), interface, *scalars.values())):
        raise ValueError("MYNN condensation inputs must be finite")
    if np.any(columns["dz"] <= 0.0) or np.any(columns["p"] <= 0.0):
        raise ValueError("MYNN condensation dz and p must be positive")

    outputs = {
        name: np.zeros((ncol, nz), dtype=np.float32)
        for name in ("qc_bl", "qi_bl", "cldfra")
    }
    outputs["vt"] = np.array(columns["vt"], dtype=np.float32, copy=True)
    outputs["vq"] = np.array(columns["vq"], dtype=np.float32, copy=True)
    outputs["sgm"] = np.array(columns["sgm"], dtype=np.float32, copy=True)
    for column in range(ncol):
        dz = columns["dz"][column]
        th = columns["th"][column]
        qw = columns["qw"][column]
        qc = columns["qc"][column]
        qi = columns["qi"][column]
        qs = columns["qs"][column]
        p = columns["p"][column]
        exner = columns["exner"][column]
        qsq = columns["qsq"][column]
        rstoch = columns["rstoch"][column]
        qc_bl = outputs["qc_bl"][column]
        qi_bl = outputs["qi_bl"][column]
        cldfra = outputs["cldfra"][column]
        vt = outputs["vt"][column]
        vq = outputs["vq"][column]
        sgm = outputs["sgm"][column]
        xland = F(scalars["xland"][column])
        k_tropo = _condensation_tropopause(th, p, nz)

        pblh2 = max(F(10.0), F(scalars["pblh"][column]))
        zagl = F(0.0)
        dzm1 = F(0.0)
        for k in range(nz - 1):
            zagl = F(zagl + F(F(0.5) * F(dz[k] + dzm1)))
            dzm1 = F(dz[k])

            t = F(th[k] * exner[k])
            xl = mynn_xl_blend(t)
            qsat_tk = mynn_qsat_blend(t, p[k])
            rh = max(
                min(RHMAX, F(qw[k] / max(F(1.0e-10), qsat_tk))), F(0.001)
            )

            # WRF stores alp(k) and bet(k) here, but the CASE(2) branch never
            # reads either one back; only a(k) and b(k) reach the outputs.
            dqsl = F(F(F(qsat_tk * EP_2) * XLV) / F(RD * F(t * t)))
            alp = F(F(1.0) / F(F(1.0) + F(dqsl * XLVCP)))
            bet = F(dqsl * exner[k])
            del alp, bet
            rsl = F(F(xl * qsat_tk) / F(RV * F(t * t)))
            cpm = F(CP + F(qw[k] * CPV))
            a = F(F(1.0) / F(F(1.0) + F(F(xl * rsl) / cpm)))
            b = F(a * rsl)

            qw_pert = F(qw[k] + F(F(F(qw[k] * F(0.5)) * rstoch[k])
                                  * F(float(spp_pbl))))
            qmq = F(qw_pert - qsat_tk)

            # r3sq is DOUBLE PRECISION in the Fortran, so the standard
            # deviation is a double-precision square root of an FP32 value
            # that is only rounded back to FP32 on assignment.
            r3sq = np.float64(max(F(qsq[k]), F(0.0)))
            sgm_k = F(np.sqrt(r3sq))
            sgm_k = min(sgm_k, F(qsat_tk * F(0.666)))
            wt = F(max(F(F(500.0) - max(F(dz[k] - F(100.0)), F(0.0))),
                       F(0.0)) / F(500.0))
            sgm_k = F(sgm_k + F(F(sgm_k * F(0.2)) * F(F(1.0) - wt)))
            qpct = F(F(QPCT_PBL * wt) + F(QPCT_TRP * F(F(1.0) - wt)))
            qpct = min(qpct, max(QPCT_SFC, F(F(QPCT_PBL * zagl) / F(500.0))))
            sgm_k = max(sgm_k, F(qsat_tk * qpct))
            sgm[k] = sgm_k

            q1 = F(qmq / sgm_k)
            wt2 = min(F(max(F(zagl - pblh2), F(0.0)) / F(300.0)), F(1.0))
            frozen = F(qi[k] + qs[k])
            if frozen > F(1.0e-9) and zagl > pblh2:
                rh_hack = min(RHMAX, F(RHCRIT + F(F(wt2 * F(0.045)) * F(
                    F(9.0) + _log10f(frozen)
                ))))
                rh = max(rh, rh_hack)
                q1_rh = F(F(-3.0) + F(F(F(3.0) * F(rh - RHCRIT))
                                      / F(F(1.0) - RHCRIT)))
                q1 = max(q1_rh, q1)
            if qc[k] > F(1.0e-6) and zagl > pblh2:
                rh_hack = min(RHMAX, F(RHCRIT + F(F(wt2 * F(0.08)) * F(
                    F(6.0) + _log10f(F(qc[k]))
                ))))
                rh = max(rh, rh_hack)
                q1_rh = F(F(-3.0) + F(F(F(3.0) * F(rh - RHCRIT))
                                      / F(F(1.0) - RHCRIT)))
                q1 = max(q1_rh, q1)

            q1k = q1
            cldfra_k = max(F(0.0), min(F(1.0), F(
                F(0.5) + F(F(0.36) * _atanf(F(F(1.8) * F(q1 + F(0.2)))))
            )))

            maxqc = max(F(qw[k] - qsat_tk), F(0.0))
            if q1k < F(0.0):
                ql_water = F(sgm_k * F(_expf(F(F(F(1.2) * q1k) - F(1.0)))))
                ql_ice = ql_water
            elif q1k > F(2.0):
                ql_water = min(F(sgm_k * q1k), maxqc)
                ql_ice = F(sgm_k * q1k)
            else:
                shape = F(sgm_k * F(
                    F(EXP_M1 + F(F(0.66) * q1k)) + F(F(0.086) * F(q1k * q1k))
                ))
                ql_water = min(shape, maxqc)
                ql_ice = shape
            if cldfra_k < F(0.001):
                ql_ice = F(0.0)
                ql_water = F(0.0)
                cldfra_k = F(0.0)

            liq_frac = min(F(1.0), max(F(0.0), F(
                F(t - TICE) / F(TLIQ - TICE)
            )))
            qc_bl[k] = F(liq_frac * ql_water)
            qi_bl[k] = F(F(F(1.0) - liq_frac) * ql_ice)
            if k + 1 >= k_tropo:
                cldfra_k = F(0.0)
                qc_bl[k] = F(0.0)
                qi_bl[k] = F(0.0)

            if F(xland - F(1.5)) >= F(0.0):
                q1k = max(q1, F(-2.5))
            else:
                q1k = max(q1, F(-2.0))
            if q1k >= F(1.0):
                fng = F(1.0)
            elif q1k >= F(-1.7):
                fng = F(_expf(F(F(-0.4) * F(q1k - F(1.0)))))
            elif q1k >= F(-2.5):
                fng = F(F(3.0) + F(_expf(F(F(-3.8) * F(q1k + F(1.7))))))
            else:
                fng = min(
                    F(F(23.9) + F(_expf(F(F(-1.6) * F(q1k + F(2.5)))))),
                    F(60.0),
                )

            cfmax = min(cldfra_k, F(0.6))
            zsl = min(max(F(25.0), F(F(0.1) * pblh2)), F(100.0))
            wt = min(F(zagl / zsl), F(1.0))
            cfmax = F(cfmax * wt)

            bb = F(F(b * t) / th[k])
            qww = F(F(1.0) + F(F(0.61) * qw[k]))
            alpha = F(F(0.61) * th[k])
            beta = F(F(F(th[k] / t) * F(xl / CP)) - F(F(1.61) * th[k]))
            vt[k] = F(F(qww - F(F(F(cfmax * beta) * bb) * fng)) - F(1.0))
            vq[k] = F(F(alpha + F(F(F(cfmax * beta) * a) * fng)) - TV0)

            fac_damp = min(F(zagl * F(0.0025)), F(1.0))
            excess = F(max(F(0.0), F(rh - F(0.92))) / F(0.145))
            cld_factor = F(F(1.0)
                           + F(fac_damp * min(F(excess * excess), F(0.37))))
            cldfra[k] = min(F(1.0), F(cld_factor * cldfra_k))

        vt[nz - 1] = vt[nz - 2]
        vq[nz - 1] = vq[nz - 2]
        qc_bl[nz - 1] = F(0.0)
        qi_bl[nz - 1] = F(0.0)
        cldfra[nz - 1] = F(0.0)
    return outputs


def mynn_retrieve_exchange_coeffs(
    values: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Translate WRF ``retrieve_exchange_coeffs`` (module_bl_mynn.F:5358).

    ``dfm``/``dfh`` are diffusivities already divided by the interface
    thickness, so this multiplies them back up.  The surface level is set to
    zero, not computed, exactly as the Fortran does.
    """

    names = ("dz", "dfm", "dfh")
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(
            f"missing MYNN exchange-coefficient inputs: {', '.join(missing)}"
        )
    columns = {
        name: np.asarray(values[name], dtype=np.float32) for name in names
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError(
            "MYNN exchange-coefficient columns must share shape (ncol,nz)"
        )
    _, nz = next(iter(shapes))
    if nz < 2:
        raise ValueError("MYNN exchange coefficients require nz >= 2")
    dz = columns["dz"]
    k_m = np.zeros_like(dz)
    k_h = np.zeros_like(dz)
    dzk = (F(0.5) * (dz[:, 1:] + dz[:, :-1])).astype(np.float32)
    k_m[:, 1:] = (columns["dfm"][:, 1:] * dzk).astype(np.float32)
    k_h[:, 1:] = (columns["dfh"][:, 1:] * dzk).astype(np.float32)
    return {"k_m": k_m, "k_h": k_h}


def _moisture_check_column_fp32(
    delt: np.float32,
    dp: np.ndarray,
    exner: np.ndarray,
    qv: np.ndarray,
    qc: np.ndarray,
    qi: np.ndarray,
    qs: np.ndarray,
    th: np.ndarray,
    dqv: np.ndarray,
    dqc: np.ndarray,
    dqi: np.ndarray,
    dqs: np.ndarray,
    dth: np.ndarray,
) -> None:
    """Translate WRF ``moisture_check`` (module_bl_mynn.F:5137-5224) in place.

    This is a borrow-from-below repair, not a clip.  A condensate deficit is
    paid for out of the vapour in the same layer (and warms it); a vapour
    deficit is paid for out of the layer *below*, weighted by the pressure
    thickness ratio; and a residual deficit in the bottom layer is spread
    proportionally over every layer that still holds more than ``2*qvmin``.
    Replacing any of that with a clamp changes the column water budget and
    therefore the trajectory.
    """

    nz = qv.size
    dqv2 = F(0.0)
    for k in range(nz - 1, -1, -1):
        dqc2 = max(F(0.0), F(QCMIN - qc[k]))
        dqi2 = max(F(0.0), F(QIMIN - qi[k]))
        dqs2 = max(F(0.0), F(QIMIN - qs[k]))
        xlvcp_ex = F(XLVCP / exner[k])
        xlscp_ex = F(XLSCP / exner[k])
        dqc[k] = F(dqc[k] + F(dqc2 / delt))
        dqi[k] = F(dqi[k] + F(dqi2 / delt))
        dqs[k] = F(dqs[k] + F(dqs2 / delt))
        dqv[k] = F(dqv[k] - F(F(F(dqc2 + dqi2) + dqs2) / delt))
        dth[k] = F(
            F(dth[k] + F(xlvcp_ex * F(dqc2 / delt)))
            + F(xlscp_ex * F(F(dqi2 + dqs2) / delt))
        )
        qc[k] = F(qc[k] + dqc2)
        qi[k] = F(qi[k] + dqi2)
        qs[k] = F(qs[k] + dqs2)
        qv[k] = F(F(F(qv[k] - dqc2) - dqi2) - dqs2)
        th[k] = F(
            F(th[k] + F(xlvcp_ex * dqc2)) + F(xlscp_ex * F(dqi2 + dqs2))
        )
        dqv2 = max(F(0.0), F(QVMIN - qv[k]))
        dqv[k] = F(dqv[k] + F(dqv2 / delt))
        qv[k] = F(qv[k] + dqv2)
        if k != 0:
            borrow = F(F(dqv2 * dp[k]) / dp[k - 1])
            qv[k - 1] = F(qv[k - 1] - borrow)
            dqv[k - 1] = F(dqv[k - 1] - F(borrow / delt))
        qv[k] = max(qv[k], QVMIN)
        qc[k] = max(qc[k], QCMIN)
        qi[k] = max(qi[k], QIMIN)
        qs[k] = max(qs[k], QIMIN)
    if dqv2 > F(1.0e-20):
        total = F(0.0)
        for k in range(nz):
            if qv[k] > F(F(2.0) * QVMIN):
                total = F(total + F(qv[k] * dp[k]))
        aa = F(F(dqv2 * dp[0]) / max(F(1.0e-20), total))
        if aa < F(0.5):
            for k in range(nz):
                if qv[k] > F(F(2.0) * QVMIN):
                    dum = F(aa * qv[k])
                    qv[k] = F(qv[k] - dum)
                    dqv[k] = F(dqv[k] - F(dum / delt))


def mynn_moisture_check(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Column-wise wrapper around WRF ``moisture_check``.

    Every array is copied first, so the caller's state is never mutated; the
    repaired species, potential temperature, and tendencies are returned.
    """

    names = (
        "dp", "exner", "qv", "qc", "qi", "qs", "th",
        "dqv", "dqc", "dqi", "dqs", "dth",
    )
    missing = [name for name in names if name not in values]
    if missing:
        raise TypeError(
            f"missing MYNN moisture-check inputs: {', '.join(missing)}"
        )
    columns = {
        name: np.array(values[name], dtype=np.float32, copy=True)
        for name in names
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError(
            "MYNN moisture-check columns must share shape (ncol,nz)"
        )
    ncol, _ = next(iter(shapes))
    if "delt" not in values:
        raise TypeError("missing MYNN moisture-check inputs: delt")
    try:
        delt = np.broadcast_to(
            np.asarray(values["delt"], dtype=np.float32), (ncol,)
        )
    except ValueError as exc:
        raise ValueError("delt is not broadcastable to ncol") from exc
    if np.any(delt <= 0.0):
        raise ValueError("MYNN moisture-check delt must be positive")
    if np.any(columns["dp"] <= 0.0):
        raise ValueError("MYNN moisture-check dp must be positive")
    for column in range(ncol):
        _moisture_check_column_fp32(
            F(delt[column]),
            *[columns[name][column] for name in names],
        )
    return columns


def _tendency_flag_identity(
    flag_qc: bool,
    flag_qi: bool,
    flag_qs: bool,
    flag_qnc: bool,
    flag_qni: bool,
    flag_qnwfa: bool,
    flag_qnifa: bool,
    flag_qnbca: bool,
    flag_ozone: bool,
    bl_mynn_mixscalars: int = 0,
) -> None:
    """Admit WRF's snow flag and reject every still-unported species flag.

    WRF derives ``FLAG_QS`` from Registry ``F_QS`` and sets it for
    ``mp_physics`` 6, 8, 10, and 18.  The flag enables the real snow column in
    ``mym_condensation``.  WRF still passes ``kzero`` to
    ``mynn_tendencies`` (``module_bl_mynn.F:1240-1242``), where snow mixing is
    also hard-disabled at ``:4618``; consequently either boolean value is
    valid here and does not alter the tendency solve itself.
    """

    if flag_qc is not True or flag_qi is not True:
        raise ValueError("MYNN tendency lane requires FLAG_QC and FLAG_QI")
    if type(flag_qs) is not bool:
        raise TypeError("MYNN tendency lane requires FLAG_QS boolean")
    # W4 mixscalars admission (this wave): with bl_mynn_mixscalars=1 the
    # five qn-family flags are REQUIRED true: the anchored fixture family
    # (w4-oracle-fixtures) pins exactly that combo, and a
    # false flag names a column the solve would read and the
    # state does not allocate.  With
    # mixscalars=0 the pre-admission refusal stands unchanged.
    qn_flags = (
        ("FLAG_QNC", flag_qnc), ("FLAG_QNI", flag_qni),
        ("FLAG_QNWFA", flag_qnwfa), ("FLAG_QNIFA", flag_qnifa),
        ("FLAG_QNBCA", flag_qnbca),
    )
    if bl_mynn_mixscalars == 1:
        for name, flag in qn_flags:
            if flag is not True:
                raise ValueError(
                    f"MYNN bl_mynn_mixscalars=1 mixes all five qn "
                    f"columns and {name} says its column is absent. "
                    "Select mp_physics=28, which carries the family, or "
                    "bl_mynn_mixscalars=0."
                )
    else:
        for name, flag in qn_flags:
            if flag is not False:
                raise ValueError(f"MYNN tendency lane requires {name} false")
    if flag_ozone is not False:
        raise ValueError("MYNN tendency lane requires FLAG_OZONE false")


def _tendency_arrays(
    values: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
           dict[str, np.ndarray], int, int]:
    """Shape-check and cast the tendency inputs shared by both lanes."""

    missing = [name for name in MYNN_TENDENCIES_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN tendency inputs: {', '.join(missing)}")

    columns = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in MYNN_TENDENCIES_LAYER_INPUTS
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError("MYNN tendency columns must share shape (ncol,nz)")
    ncol, nz = next(iter(shapes))
    if nz < 3:
        raise ValueError("MYNN tendencies require nz >= 3")
    interfaces = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in MYNN_TENDENCIES_INTERFACE_INPUTS
    }
    bad_shape = [
        name for name, array in interfaces.items()
        if array.shape != (ncol, nz + 1)
    ]
    if bad_shape:
        raise ValueError(
            "MYNN tendency mass-flux inputs must have shape (ncol,nz+1): "
            + ", ".join(sorted(bad_shape))
        )
    scalars: dict[str, np.ndarray] = {}
    for name in MYNN_TENDENCIES_SCALAR_INPUTS:
        try:
            scalars[name] = np.broadcast_to(
                np.asarray(values[name], dtype=np.float32), (ncol,)
            )
        except ValueError as exc:
            raise ValueError(f"{name} is not broadcastable to ncol") from exc
    if any(not np.isfinite(array).all() for array in
           (*columns.values(), *interfaces.values(), *scalars.values())):
        raise ValueError("MYNN tendency inputs must be finite")
    if np.any(columns["dz"] <= 0.0) or np.any(columns["rho"] <= 0.0):
        raise ValueError("MYNN tendency dz and rho must be positive")
    if np.any(columns["exner"] <= 0.0) or np.any(columns["p"] <= 0.0):
        raise ValueError("MYNN tendency exner and p must be positive")
    if np.any(scalars["delt"] <= 0.0) or np.any(scalars["wspd"] <= 0.0):
        raise ValueError("MYNN tendency delt and wspd must be positive")
    if np.any(scalars["psfc"] <= 0.0):
        raise ValueError("MYNN tendency psfc must be positive")
    return columns, interfaces, scalars, ncol, nz


def _tendency_solve(
    columns: Mapping[str, np.ndarray],
    interfaces: Mapping[str, np.ndarray],
    scalars: Mapping[str, np.ndarray],
    ncol: int,
    nz: int,
    onoff: np.float32,
    mixscalars: bool = False,
    surface_density_tv: bool = False,
) -> dict[str, np.ndarray]:
    """The per-column body of ``mynn_tendencies``.

    ``onoff`` is the Fortran factor built at ``module_bl_mynn.F:4130-4134``
    from ``bl_mynn_edmf_mom``.  Every other admitted-identity decision is
    already resolved by the caller.
    """

    outputs = {
        name: np.zeros((ncol, nz), dtype=np.float32)
        for name in MYNN_TENDENCIES_OUTPUTS
    }
    half = F(0.5)
    for column in range(ncol):
        dz = columns["dz"][column]
        rho = columns["rho"][column]
        u = columns["u"][column]
        v = columns["v"][column]
        th = columns["th"][column]
        tk = columns["tk"][column]
        qv = columns["qv"][column]
        p = columns["p"][column]
        exner = columns["exner"][column]
        tcd = columns["tcd"][column]
        qcd = columns["qcd"][column]
        dfm = columns["dfm"][column]
        dfh = columns["dfh"][column]
        diss_heat = columns["diss_heat"][column]
        ozone = columns["ozone"][column]
        thl = columns["thl"][column].copy()
        sqv = columns["sqv"][column]
        sqc = columns["sqc"][column]
        sqi = columns["sqi"][column]
        sqs = columns["sqs"][column]
        sub_thl = columns["sub_thl"][column]
        sub_sqv = columns["sub_sqv"][column]
        sub_u = columns["sub_u"][column]
        sub_v = columns["sub_v"][column]
        det_thl = columns["det_thl"][column]
        det_sqv = columns["det_sqv"][column]
        det_sqc = columns["det_sqc"][column]
        det_u = columns["det_u"][column]
        det_v = columns["det_v"][column]
        s_aw = interfaces["s_aw"][column]
        s_awthl = interfaces["s_awthl"][column]
        s_awqv = interfaces["s_awqv"][column]
        s_awqc = interfaces["s_awqc"][column]
        s_awu = interfaces["s_awu"][column]
        s_awv = interfaces["s_awv"][column]
        sd_aw = interfaces["sd_aw"][column]
        sd_awthl = interfaces["sd_awthl"][column]
        sd_awqv = interfaces["sd_awqv"][column]
        sd_awqc = interfaces["sd_awqc"][column]
        sd_awu = interfaces["sd_awu"][column]
        sd_awv = interfaces["sd_awv"][column]
        delt = F(scalars["delt"][column])
        psfc = F(scalars["psfc"][column])
        ust = F(scalars["ust"][column])
        wspd = F(scalars["wspd"][column])
        uoce = F(scalars["uoce"][column])
        voce = F(scalars["voce"][column])
        flt = F(scalars["flt"][column])
        flqv = F(scalars["flqv"][column])
        flqc = F(scalars["flqc"][column])

        dtz = np.empty(nz, dtype=np.float32)
        rhoinv = np.empty(nz, dtype=np.float32)
        delp = np.empty(nz, dtype=np.float32)
        rhoz = np.empty(nz + 1, dtype=np.float32)
        khdz = np.empty(nz + 1, dtype=np.float32)
        kmdz = np.empty(nz + 1, dtype=np.float32)
        if surface_density_tv:
            # Referee of mynn_fixes_v61 edit A2 (correct-mode MY1;
            # NCAR/MYNN-EDMF eb5029c): psfc/(r_d*tk(kts)*(one+p608*qv(kts))).
            rhosfc = F(psfc / F(F(RD * tk[0]) * F(F(1.0) + F(P608 * qv[0]))))
        else:
            rhosfc = F(psfc / F(RD * F(tk[0] + F(P608 * qv[0]))))
        dtz[0] = F(delt / dz[0])
        rhoz[0] = rho[0]
        rhoinv[0] = F(F(1.0) / rho[0])
        khdz[0] = F(rhoz[0] * dfh[0])
        kmdz[0] = F(rhoz[0] * dfm[0])
        delp[0] = F(
            psfc
            - F(F(F(p[1] * dz[0]) + F(p[0] * dz[1])) / F(dz[0] + dz[1]))
        )
        for k in range(1, nz):
            dtz[k] = F(delt / dz[k])
            rhoz[k] = F(
                F(F(rho[k] * dz[k - 1]) + F(rho[k - 1] * dz[k]))
                / F(dz[k - 1] + dz[k])
            )
            rhoz[k] = max(rhoz[k], F(1.0e-4))
            rhoinv[k] = F(F(1.0) / max(rho[k], F(1.0e-4)))
            khdz[k] = F(rhoz[k] * dfh[k])
            kmdz[k] = F(rhoz[k] * dfm[k])
        for k in range(1, nz - 1):
            delp[k] = F(
                F(F(F(p[k] * dz[k - 1]) + F(p[k - 1] * dz[k]))
                  / F(dz[k] + dz[k - 1]))
                - F(F(F(p[k + 1] * dz[k]) + F(p[k] * dz[k + 1]))
                    / F(dz[k] + dz[k + 1]))
            )
        delp[nz - 1] = delp[nz - 2]
        rhoz[nz] = rhoz[nz - 1]
        khdz[nz] = F(rhoz[nz] * dfh[nz - 1])
        kmdz[nz] = F(rhoz[nz] * dfm[nz - 1])
        # Stability floors for the mass flux.  Inert while s_aw is zero, but
        # they are the reason kqdz/khdz cannot simply be reused from
        # mym_predict once DMP_mf is live.
        for k in range(1, nz - 1):
            khdz[k] = max(khdz[k], F(half * s_aw[k]))
            khdz[k] = max(khdz[k], F(-F(half * F(s_aw[k] - s_aw[k + 1]))))
            kmdz[k] = max(kmdz[k], F(half * s_aw[k]))
            kmdz[k] = max(kmdz[k], F(-F(half * F(s_aw[k] - s_aw[k + 1]))))

        a = np.empty(nz, dtype=np.float32)
        b = np.empty(nz, dtype=np.float32)
        c = np.empty(nz, dtype=np.float32)
        d = np.empty(nz, dtype=np.float32)
        # 0.5*dtz(k)*rhoinv(k), the shared prefactor of every mass-flux term.
        hdz = np.asarray(
            [F(F(half * dtz[k]) * rhoinv[k]) for k in range(nz)],
            dtype=np.float32,
        )
        dzinv = np.asarray(
            [F(dtz[k] * rhoinv[k]) for k in range(nz)], dtype=np.float32
        )

        # ---- momentum ------------------------------------------------
        drag = F(F(rhosfc * F(ust * ust)) / wspd)
        a[0] = -F(F(dtz[0] * kmdz[0]) * rhoinv[0])
        b[0] = F(F(F(1.0) + F(F(dtz[0] * F(kmdz[1] + drag)) * rhoinv[0]))
                 - F(F(hdz[0] * s_aw[1]) * onoff))
        b[0] = F(b[0] - F(F(hdz[0] * sd_aw[1]) * onoff))
        c[0] = F(-F(F(dtz[0] * kmdz[1]) * rhoinv[0])
                 - F(F(hdz[0] * s_aw[1]) * onoff))
        c[0] = F(c[0] - F(F(hdz[0] * sd_aw[1]) * onoff))
        for k in range(1, nz - 1):
            a[k] = F(-F(F(dtz[k] * kmdz[k]) * rhoinv[k])
                     + F(F(hdz[k] * s_aw[k]) * onoff))
            a[k] = F(a[k] + F(F(hdz[k] * sd_aw[k]) * onoff))
            b[k] = F(F(F(1.0)
                       + F(F(dtz[k] * F(kmdz[k] + kmdz[k + 1])) * rhoinv[k]))
                     + F(F(hdz[k] * F(s_aw[k] - s_aw[k + 1])) * onoff))
            b[k] = F(b[k] + F(F(hdz[k] * F(sd_aw[k] - sd_aw[k + 1])) * onoff))
            c[k] = F(-F(F(dtz[k] * kmdz[k + 1]) * rhoinv[k])
                     - F(F(hdz[k] * s_aw[k + 1]) * onoff))
            c[k] = F(c[k] - F(F(hdz[k] * sd_aw[k + 1]) * onoff))
        a[nz - 1] = F(0.0)
        b[nz - 1] = F(1.0)
        c[nz - 1] = F(0.0)

        d[0] = F(F(u[0] + F(F(F(dtz[0] * uoce) * F(ust * ust)) / wspd))
                 - F(F(dzinv[0] * s_awu[1]) * onoff))
        d[0] = F(d[0] + F(F(dzinv[0] * sd_awu[1]) * onoff))
        d[0] = F(F(d[0] + F(sub_u[0] * delt)) + F(det_u[0] * delt))
        for k in range(1, nz - 1):
            d[k] = F(u[k]
                     + F(F(dzinv[k] * F(s_awu[k] - s_awu[k + 1])) * onoff))
            d[k] = F(d[k]
                     - F(F(dzinv[k] * F(sd_awu[k] - sd_awu[k + 1])) * onoff))
            d[k] = F(F(d[k] + F(sub_u[k] * delt)) + F(det_u[k] * delt))
        d[nz - 1] = u[nz - 1]
        x = _tridiag2_fp32(a, b, c, d)
        outputs["du"][column] = np.asarray(
            [F(F(x[k] - u[k]) / delt) for k in range(nz)], dtype=np.float32
        )

        d[0] = F(F(v[0] + F(F(F(dtz[0] * voce) * F(ust * ust)) / wspd))
                 - F(F(dzinv[0] * s_awv[1]) * onoff))
        d[0] = F(d[0] + F(F(dzinv[0] * sd_awv[1]) * onoff))
        d[0] = F(F(d[0] + F(sub_v[0] * delt)) + F(det_v[0] * delt))
        for k in range(1, nz - 1):
            d[k] = F(v[k]
                     + F(F(dzinv[k] * F(s_awv[k] - s_awv[k + 1])) * onoff))
            d[k] = F(d[k]
                     - F(F(dzinv[k] * F(sd_awv[k] - sd_awv[k + 1])) * onoff))
            d[k] = F(F(d[k] + F(sub_v[k] * delt)) + F(det_v[k] * delt))
        d[nz - 1] = v[nz - 1]
        x = _tridiag2_fp32(a, b, c, d)
        outputs["dv"][column] = np.asarray(
            [F(F(x[k] - v[k]) / delt) for k in range(nz)], dtype=np.float32
        )

        # ---- shared heat/moisture matrix ------------------------------
        a[0] = -F(F(dtz[0] * khdz[0]) * rhoinv[0])
        b[0] = F(F(F(1.0) + F(F(dtz[0] * F(khdz[1] + khdz[0])) * rhoinv[0]))
                 - F(hdz[0] * s_aw[1]))
        b[0] = F(b[0] - F(hdz[0] * sd_aw[1]))
        c[0] = F(-F(F(dtz[0] * khdz[1]) * rhoinv[0]) - F(hdz[0] * s_aw[1]))
        c[0] = F(c[0] - F(hdz[0] * sd_aw[1]))
        for k in range(1, nz - 1):
            a[k] = F(-F(F(dtz[k] * khdz[k]) * rhoinv[k])
                     + F(hdz[k] * s_aw[k]))
            a[k] = F(a[k] + F(hdz[k] * sd_aw[k]))
            b[k] = F(F(F(1.0)
                       + F(F(dtz[k] * F(khdz[k] + khdz[k + 1])) * rhoinv[k]))
                     + F(hdz[k] * F(s_aw[k] - s_aw[k + 1])))
            b[k] = F(b[k] + F(hdz[k] * F(sd_aw[k] - sd_aw[k + 1])))
            c[k] = F(-F(F(dtz[k] * khdz[k + 1]) * rhoinv[k])
                     - F(hdz[k] * s_aw[k + 1]))
            c[k] = F(c[k] - F(hdz[k] * sd_aw[k + 1]))
        a[nz - 1] = F(0.0)
        b[nz - 1] = F(1.0)
        c[nz - 1] = F(0.0)

        # ---- liquid-water potential temperature ------------------------
        d[0] = F(F(thl[0] + F(F(F(dtz[0] * rhosfc) * flt) * rhoinv[0]))
                 + F(tcd[0] * delt))
        d[0] = F(F(d[0] - F(dzinv[0] * s_awthl[1]))
                 - F(dzinv[0] * sd_awthl[1]))
        d[0] = F(F(F(d[0] + F(diss_heat[0] * delt)) + F(sub_thl[0] * delt))
                 + F(det_thl[0] * delt))
        for k in range(1, nz - 1):
            d[k] = F(thl[k] + F(tcd[k] * delt))
            d[k] = F(d[k] + F(dzinv[k] * F(s_awthl[k] - s_awthl[k + 1])))
            d[k] = F(d[k] + F(dzinv[k] * F(sd_awthl[k] - sd_awthl[k + 1])))
            d[k] = F(F(F(d[k] + F(diss_heat[k] * delt))
                       + F(sub_thl[k] * delt)) + F(det_thl[k] * delt))
        d[nz - 1] = thl[nz - 1]
        thl = _tridiag2_fp32(a, b, c, d)

        # ---- cloud water ----------------------------------------------
        d[0] = F(F(sqc[0] + F(F(F(dtz[0] * rhosfc) * flqc) * rhoinv[0]))
                 + F(qcd[0] * delt))
        d[0] = F(F(d[0] - F(dzinv[0] * s_awqc[1])) - F(dzinv[0] * sd_awqc[1]))
        d[0] = F(d[0] + F(det_sqc[0] * delt))
        for k in range(1, nz - 1):
            d[k] = F(sqc[k] + F(qcd[k] * delt))
            d[k] = F(d[k] + F(dzinv[k] * F(s_awqc[k] - s_awqc[k + 1])))
            d[k] = F(d[k] + F(dzinv[k] * F(sd_awqc[k] - sd_awqc[k + 1])))
            d[k] = F(d[k] + F(det_sqc[k] * delt))
        d[nz - 1] = sqc[nz - 1]
        sqc2 = _tridiag2_fp32(a, b, c, d)

        # ---- water vapour ----------------------------------------------
        # WRF limits an unreasonably large *negative* surface moisture flux.
        # For any positive sqv(kts) the MIN collapses to 0.0, so a downward
        # flux is not limited but deleted; this reproduces that verbatim.
        qvflux = flqv
        if qvflux < F(0.0):
            qvflux = max(
                qvflux,
                F(min(F(F(F(0.9) * sqv[0]) - F(1.0e-8)), F(0.0)) / dtz[0]),
            )
        d[0] = F(F(sqv[0] + F(F(F(dtz[0] * rhosfc) * qvflux) * rhoinv[0]))
                 + F(qcd[0] * delt))
        d[0] = F(F(d[0] - F(dzinv[0] * s_awqv[1])) - F(dzinv[0] * sd_awqv[1]))
        d[0] = F(F(d[0] + F(sub_sqv[0] * delt)) + F(det_sqv[0] * delt))
        for k in range(1, nz - 1):
            d[k] = F(sqv[k] + F(qcd[k] * delt))
            d[k] = F(d[k] + F(dzinv[k] * F(s_awqv[k] - s_awqv[k + 1])))
            d[k] = F(d[k] + F(dzinv[k] * F(sd_awqv[k] - sd_awqv[k + 1])))
            d[k] = F(F(d[k] + F(sub_sqv[k] * delt)) + F(det_sqv[k] * delt))
        d[nz - 1] = sqv[nz - 1]
        sqv2 = _tridiag2_fp32(a, b, c, d)

        # ---- cloud ice: pure diffusion, no mass flux -------------------
        a[0] = -F(F(dtz[0] * khdz[0]) * rhoinv[0])
        b[0] = F(F(1.0) + F(F(dtz[0] * F(khdz[1] + khdz[0])) * rhoinv[0]))
        c[0] = -F(F(dtz[0] * khdz[1]) * rhoinv[0])
        d[0] = sqi[0]
        for k in range(1, nz - 1):
            a[k] = -F(F(dtz[k] * khdz[k]) * rhoinv[k])
            b[k] = F(F(1.0)
                     + F(F(dtz[k] * F(khdz[k] + khdz[k + 1])) * rhoinv[k]))
            c[k] = -F(F(dtz[k] * khdz[k + 1]) * rhoinv[k])
            d[k] = sqi[k]
        a[nz - 1] = F(0.0)
        b[nz - 1] = F(1.0)
        c[nz - 1] = F(0.0)
        d[nz - 1] = sqi[nz - 1]
        sqi2 = _tridiag2_fp32(a, b, c, d)
        # Snow mixing is hard-disabled at module_bl_mynn.F:4618.
        sqs2 = sqs.copy()

        # ---- W4 mixscalars: the five stock qn solves --------------------
        # (module_bl_mynn.F:4654/:4695/:4736/:4778/:4820; tendencies
        # :4957-4966/:4998-5007/:5060-5077/:5082-5091.)  Additive arm in
        # woof.core.mynn_scalar_mix, consuming THIS solve's dtz/rhoinv/
        # floored khdz/hdz/dzinv and s_aw: nothing recomputed.  Import is
        # local so the admitted mixscalars=0 lane never touches the module.
        if mixscalars:
            from woof.core.mynn_scalar_mix import mix_scalar_column
            for species in ("qni", "qnc", "qnwfa", "qnifa", "qnbca"):
                _, dqn = mix_scalar_column(
                    columns[species][column],
                    dtz, rhoinv, khdz, hdz, dzinv,
                    s_aw, interfaces[f"s_aw{species}"][column], delt,
                )
                outputs[f"d{species}"][column] = dqn

        dqv = np.asarray(
            [F(F(sqv2[k] - sqv[k]) / delt) for k in range(nz)],
            dtype=np.float32,
        )
        dqc = np.asarray(
            [F(F(sqc2[k] - sqc[k]) / delt) for k in range(nz)],
            dtype=np.float32,
        )
        dqi = np.asarray(
            [F(F(sqi2[k] - sqi[k]) / delt) for k in range(nz)],
            dtype=np.float32,
        )
        dqs = np.zeros(nz, dtype=np.float32)
        # dth is zeroed at module_bl_mynn.F:4173 so moisture_check can
        # accumulate into it; the theta block below then overwrites it, but
        # moisture_check's mutation of thl, sqc2 and sqi2 still reaches the
        # answer.  That asymmetry is WRF's, not a transcription slip.
        dth = np.zeros(nz, dtype=np.float32)
        _moisture_check_column_fp32(
            delt, delp, exner, sqv2, sqc2, sqi2, sqs2, thl,
            dqv, dqc, dqi, dqs, dth,
        )

        dozone = np.zeros(nz, dtype=np.float32)
        for k in range(nz):
            if F(F(dozone[k] * delt) + ozone[k]) < F(0.0):
                dozone[k] = F(-F(ozone[k] * F(0.99)) / delt)

        dth = np.asarray([
            F(F(F(F(thl[k] + F(F(XLVCP / exner[k]) * sqc2[k]))
                  + F(F(XLSCP / exner[k]) * sqi2[k])) - th[k]) / delt)
            for k in range(nz)
        ], dtype=np.float32)

        outputs["dth"][column] = dth
        outputs["dqv"][column] = dqv
        outputs["dqc"][column] = dqc
        outputs["dqi"][column] = dqi
        outputs["dqs"][column] = dqs
        outputs["dozone"][column] = dozone
        outputs["thl"][column] = thl
    return outputs


def mynn_tendencies_nomf(
    values: Mapping[str, object],
    *,
    bl_mynn_cloudmix: int = 1,
    bl_mynn_mixqt: int = 0,
    bl_mynn_edmf: int = 0,
    bl_mynn_edmf_mom: int = 0,
    bl_mynn_mixscalars: int = 0,
    flag_qc: bool = True,
    flag_qi: bool = True,
    flag_qs: bool = False,
    flag_qnc: bool = False,
    flag_qni: bool = False,
    flag_qnwfa: bool = False,
    flag_qnifa: bool = False,
    flag_qnbca: bool = False,
    flag_ozone: bool = False,
) -> dict[str, np.ndarray]:
    """Translate WRF ``mynn_tendencies`` with the mass flux zeroed.

    Pinned identity: ``bl_mynn_cloudmix=1``, ``bl_mynn_mixqt=0``,
    ``bl_mynn_edmf=0``, ``bl_mynn_edmf_mom=0``, ``bl_mynn_mixscalars=0``,
    ``FLAG_QC``/``FLAG_QI`` true and every other species flag false.  Under
    that identity the active solves are u, v, thl, sqc, sqv, and sqi; total
    water is not mixed, snow mixing is hard-disabled at
    ``module_bl_mynn.F:4618`` by an ``.AND. .false.``, and the number
    concentration, aerosol, black-carbon, and ozone systems all alias their
    inputs, leaving zero tendencies.

    ``bl_mynn_edmf_mom=0`` sets the Fortran ``onoff`` factor to zero, which
    removes the mass flux from the momentum systems only.  The thl, sqc, and
    sqv systems take ``s_aw``/``sd_aw`` unconditionally, so this lane also
    requires every ``s_aw*``, ``sd_aw*``, ``sub_*``, and ``det_*`` input to
    be identically zero and rejects anything else rather than producing an
    answer this fixture never measured.  Use ``mynn_tendencies_default`` for
    the mass-flux-admitted lane.

    Returns the tendencies WRF exports plus the updated ``thl``.  ``sqv2``,
    ``sqc2``, and ``sqi2`` stay local in the Fortran, so they are not
    outputs here either.
    """

    if bl_mynn_cloudmix != 1 or type(bl_mynn_cloudmix) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_cloudmix=1")
    if bl_mynn_mixqt != 0 or type(bl_mynn_mixqt) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_mixqt=0")
    if bl_mynn_edmf != 0 or type(bl_mynn_edmf) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_edmf=0")
    if bl_mynn_edmf_mom != 0 or type(bl_mynn_edmf_mom) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_edmf_mom=0")
    # Not a missing measurement: the no-mass-flux lane never binds the
    # five qn columns or their s_awqn* interfaces, so there is nothing
    # for the mixscalars arms to solve here.
    if bl_mynn_mixscalars != 0 or type(bl_mynn_mixscalars) is not int:
        raise ValueError(
            "MYNN no-mass-flux tendency lane carries no qn columns, so "
            "bl_mynn_mixscalars=1 has nothing to mix here. Run the "
            "mass-flux lane (bl_mynn_edmf=1) or set "
            "bl_mynn_mixscalars=0.")
    _tendency_flag_identity(
        flag_qc, flag_qi, flag_qs, flag_qnc, flag_qni,
        flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone,
    )
    columns, interfaces, scalars, ncol, nz = _tendency_arrays(values)
    forced = [
        name for name in (
            *MYNN_TENDENCIES_INTERFACE_INPUTS,
            "sub_thl", "sub_sqv", "sub_u", "sub_v",
            "det_thl", "det_sqv", "det_sqc", "det_u", "det_v",
        )
        if np.any((interfaces.get(name, columns.get(name))) != 0.0)
    ]
    if forced:
        raise ValueError(
            "this MYNN tendency lane admits only zero mass-flux, subsidence "
            "and detrainment forcing; nonzero: " + ", ".join(sorted(forced))
        )
    return _tendency_solve(
        columns, interfaces, scalars, ncol, nz, F(0.0)
    )


def mynn_tendencies_default(
    values: Mapping[str, object],
    *,
    bl_mynn_cloudmix: int = 1,
    bl_mynn_mixqt: int = 0,
    bl_mynn_edmf: int = 1,
    bl_mynn_edmf_mom: int = 1,
    bl_mynn_mixscalars: int = 0,
    flag_qc: bool = True,
    flag_qi: bool = True,
    flag_qs: bool = False,
    flag_qnc: bool = False,
    flag_qni: bool = False,
    flag_qnwfa: bool = False,
    flag_qnifa: bool = False,
    flag_qnbca: bool = False,
    flag_ozone: bool = False,
    surface_density_tv: bool = False,
) -> dict[str, np.ndarray]:
    """Translate WRF ``mynn_tendencies`` with the mass flux admitted.

    ``surface_density_tv`` is the referee of ``mynn_fixes_v61`` edit A2
    (correct-mode MY1): the surface density from the virtual temperature
    (NCAR/MYNN-EDMF eb5029c) instead of WRF v4.6.1's
    ``psfc/(r_d*(tk + p608*qv))``.  Off (the default) is WRF's referee.

    Same routine as :func:`mynn_tendencies_nomf` and the same admitted option
    identity except that ``s_aw*``, ``sd_aw*``, ``sub_*`` and ``det_*`` may be
    nonzero, which is what ``DMP_mf`` produces once ``bl_mynn_edmf>0``.

    Three consequences of admitting the flux are worth naming, because they
    are all invisible while the forcing is zero:

    * ``bl_mynn_edmf_mom`` selects the Fortran ``onoff`` factor
      (``module_bl_mynn.F:4130-4134``), which multiplies the mass flux in the
      u and v systems only.  The thl, sqc and sqv systems take ``s_aw*`` and
      ``sd_aw*`` unconditionally, so ``bl_mynn_edmf_mom=0`` does not turn the
      mass flux off, it turns off *momentum* transport by the mass flux.
    * the ``MAX`` floors at ``module_bl_mynn.F:4163-4169`` raise ``khdz`` and
      ``kmdz`` toward the mass flux for numerical stability.  They bind in
      practice: the ``deep_plume`` fixture column trips seven ``khdz`` and six
      ``kmdz`` floors.
    * ``bl_mynn_edmf`` is declared ``intent(in)`` at ``:4070-4072`` and is
      then *never read* in the body of ``mynn_tendencies``.  It is accepted
      here for call-site symmetry with WRF and validated, but it selects
      nothing; the mass flux enters solely through the arrays.

    ``sd_aw*`` (``bl_mynn_edmf_dd=0`` at ``:330``) and ``sub_*``/``det_*``
    (``env_subs=.false.`` at ``:336``) are always zero when the WRF driver
    calls this routine, but they are unconditional terms in the transcribed
    arithmetic, so the oracle exercises them with explicitly labelled probe
    columns rather than leaving them unmeasured.
    """

    if bl_mynn_cloudmix != 1 or type(bl_mynn_cloudmix) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_cloudmix=1")
    if bl_mynn_mixqt != 0 or type(bl_mynn_mixqt) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_mixqt=0")
    if bl_mynn_edmf not in (0, 1) or type(bl_mynn_edmf) is not int:
        raise ValueError("MYNN tendency lane requires bl_mynn_edmf in {0,1}")
    if bl_mynn_edmf_mom not in (0, 1) or type(bl_mynn_edmf_mom) is not int:
        raise ValueError(
            "MYNN tendency lane requires bl_mynn_edmf_mom in {0,1}"
        )
    # W4 mixscalars admission (this wave; anchored fixtures
    # w4-oracle-fixtures): bl_mynn_mixscalars=1 routes the
    # five stock qn solves through woof.core.mynn_scalar_mix.  Any other
    # other value has no meaning: WRF defines the key at 0 or 1.
    if bl_mynn_mixscalars not in (0, 1) or \
            type(bl_mynn_mixscalars) is not int:
        raise ValueError(
            "MYNN tendency lane requires bl_mynn_mixscalars in {0,1}"
        )
    _tendency_flag_identity(
        flag_qc, flag_qi, flag_qs, flag_qnc, flag_qni,
        flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone,
        bl_mynn_mixscalars,
    )
    columns, interfaces, scalars, ncol, nz = _tendency_arrays(values)
    onoff = F(0.0) if bl_mynn_edmf_mom == 0 else F(1.0)
    if bl_mynn_mixscalars == 1:
        missing = [
            name for name in (*MYNN_TENDENCIES_QN_LAYER_INPUTS,
                              *MYNN_TENDENCIES_QN_INTERFACE_INPUTS)
            if name not in values
        ]
        if missing:
            raise TypeError(
                "missing MYNN mixscalars inputs: " + ", ".join(missing)
            )
        for name in MYNN_TENDENCIES_QN_LAYER_INPUTS:
            array = np.asarray(values[name], dtype=np.float32)
            if array.shape != (ncol, nz):
                raise ValueError(f"{name} must have shape (ncol,nz)")
            columns[name] = array
        for name in MYNN_TENDENCIES_QN_INTERFACE_INPUTS:
            array = np.asarray(values[name], dtype=np.float32)
            if array.shape != (ncol, nz + 1):
                raise ValueError(f"{name} must have shape (ncol,nz+1)")
            interfaces[name] = array
    return _tendency_solve(columns, interfaces, scalars, ncol, nz, onoff,
                           mixscalars=(bl_mynn_mixscalars == 1),
                           surface_density_tv=surface_density_tv)


__all__ = [
    "MYNN_CONDENSATION_INPUTS",
    "MYNN_CONDENSATION_OUTPUTS",
    "MYNN_LEVEL2_INPUTS",
    "MYNN_LEVEL2_OUTPUTS",
    "MYNN_MIXLENGTH_INPUTS",
    "MYNN_PREDICT_INPUTS",
    "MYNN_TENDENCIES_INPUTS",
    "MYNN_TENDENCIES_INTERFACE_INPUTS",
    "MYNN_TENDENCIES_LAYER_INPUTS",
    "MYNN_TENDENCIES_OUTPUTS",
    "MYNN_TENDENCIES_SCALAR_INPUTS",
    "MYNN_TURBULENCE_INPUTS",
    "mynn_condensation_default",
    "mynn_esat_blend",
    "mynn_get_pblh",
    "mynn_level2_pairs",
    "mynn_mixlength_default",
    "mynn_moisture_check",
    "mynn_predict_default",
    "mynn_qsat_blend",
    "mynn_retrieve_exchange_coeffs",
    "mynn_tendencies_default",
    "mynn_tendencies_nomf",
    "mynn_turbulence_default",
    "mynn_scale_aware",
    "mynn_xl_blend",
]


# DMP_mf argument groups under the admitted identity.  WRF also declares dt,
# ust, kpbl, flqv, qke, qnc, qni, qnwfa, qnifa, qnbca, sgm, qc_bl1d_old,
# cldfra_bl1d_old, nchem/chem1 and the seven F_Q* flags, and none of them can
# reach an output: dt appears only inside the env_subs block, where it gates
# and scales the subsidence limiter at module_bl_mynn.F:6568-6569; qke and the
# five number-concentration columns only feed
# UPQKE/UPQNC/UPQNI/UPQNWFA/UPQNIFA/UPQNBCA, which are accumulated only under
# tke_opt>0 (:6429) and scalar_opt>0 (:6447); sgm survives only in comments,
# the sigq blend at :6684 and the debug print at :6714; and ust, kpbl, flqv,
# qc_bl1d_old, cldfra_bl1d_old and the F_Q* flags are never referenced in the
# body at all -- ust reaches only the commented print at :5870.
# The oracle's dead_probe case is the same column as land_cumulus with every
# one of them moved off its baseline, and every output column matches bit for
# bit, so the exclusion is demonstrated rather than argued.
MYNN_DMP_MF_COLUMN_INPUTS = (
    "dz", "p", "rho", "u", "v", "w", "th", "thl", "thv", "tk",
    "qt", "qv", "qc", "exner", "rstoch", "qc_bl", "cldfra_bl", "vt", "vq",
)
MYNN_DMP_MF_SCALAR_INPUTS = (
    "flt", "fltv", "flq", "pblh", "dx", "landsea", "ts", "psig_shcu",
)
MYNN_DMP_MF_INPUTS = (
    *MYNN_DMP_MF_COLUMN_INPUTS, "zw", *MYNN_DMP_MF_SCALAR_INPUTS,
)
MYNN_DMP_MF_LAYER_OUTPUTS = (
    "edmf_a", "edmf_w", "edmf_qt", "edmf_thl", "edmf_ent", "edmf_qc",
    "qc_bl", "cldfra_bl", "vt", "vq",
)
MYNN_DMP_MF_INTERFACE_OUTPUTS = (
    "s_aw", "s_awthl", "s_awqt", "s_awqv", "s_awqc", "s_awu", "s_awv",
)
# env_subs is .false. at module_bl_mynn.F:336 and bl_mynn_edmf_dd is 0
# at :330, so the whole subsidence and dynamic-detrainment block
# (module_bl_mynn.F:6547-6617) never executes and these nine tendencies leave
# DMP_mf exactly as it zeroed them at :5926-5934 (:5925 is the comment that
# heads that block, not the first assignment).
MYNN_DMP_MF_ZERO_OUTPUTS = (
    "sub_thl", "sub_sqv", "sub_u", "sub_v",
    "det_thl", "det_sqv", "det_sqc", "det_u", "det_v",
)
# tke_opt=0 and scalar_opt=0 leave these interface fluxes at their zeroed
# state as well; mix_chem is false, so s_awchem never exists.
MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS = (
    "s_awqke", "s_awqnc", "s_awqni", "s_awqnwfa", "s_awqnifa", "s_awqnbca",
)
# W4 mixscalars admission (this wave): with bl_mynn_mixscalars=1 the five
# s_awqn* names above stop being structural zeros: mynn_dmp_mf overwrites
# them with the live :6447-6456 accumulation.  s_awqke keeps its tke_opt=0
# structural zero either way.
MYNN_DMP_MF_QN_COLUMN_INPUTS = ("qnc", "qni", "qnwfa", "qnifa", "qnbca")
MYNN_DMP_MF_SCALAR_OUTPUTS = ("maxwidth", "ktop", "ztop", "maxmf")
# module_bl_mynn.F:5758 nup, :5781-5783 Wa/Wb/Wc (only used by the
# commented-out StEM forms), :5792-5798 Atot/lmax/lmin/dcut, :5824
# cf_thresh, :5838 fluxportion, :5851 Cdet, :5857 Csub, :5860 pgfac.
DMP_NUP = 8
DMP_ATOT = F(0.10)
DMP_LMAX = F(1000.0)
DMP_LMIN = F(300.0)
DMP_DLMIN = F(0.0)
DMP_DCUT = F(1.2)
DMP_FLUXPORTION = F(0.75)
DMP_CF_THRESH = F(0.5)
DMP_D = F(-1.9)
GRAV = F(9.81)
RVOVRD = F(RV / RD)
RCP = F(RD / CP)
P1000MB = F(100000.0)


def _up(a_k, a_k1, dz_k, dz_k1) -> np.float32:
    """WRF's interface interpolation of a layer field.

    ``(a_k*dz_k1 + a_k1*dz_k) / (dz_k1 + dz_k)``.
    """

    return F(F(F(a_k * dz_k1) + F(a_k1 * dz_k)) / F(dz_k1 + dz_k))


def _condensation_edmf(
    qt: np.float32, thl: np.float32, p: np.float32, zagl: np.float32,
    qc: np.float32,
) -> tuple[np.float32, np.float32]:
    """Translate ``module_bl_mynn.F:6827-6884`` condensation_edmf.

    ``qc`` is intent(inout) in the Fortran: the plume carries its condensate
    up as the first guess, which is why the loop is not restarted from zero.
    Returns ``(thv, qc)``.
    """

    exn = _powf(F(p / P1000MB), RCP)
    for _ in range(50):
        t = F(F(exn * thl) + F(XLVCP * qc))
        qs = mynn_qsat_blend(t, p)
        qcold = qc
        qc = F(F(F(0.5) * qc) + F(F(0.5) * max(F(qt - qs), F(0.0))))
        if abs(F(qc - qcold)) < F(1.0e-6):
            break
    t = F(F(exn * thl) + F(XLVCP * qc))
    qs = mynn_qsat_blend(t, p)
    qc = max(F(qt - qs), F(0.0))
    if zagl < F(100.0):
        qc = F(0.0)
    thv = F(F(thl + F(XLVCP * qc))
            * F(F(F(1.0) + F(qt * F(RVOVRD - F(1.0)))) - F(RVOVRD * qc)))
    return thv, qc


def _dmp_mf_column(
    dz: np.ndarray, p: np.ndarray, rho: np.ndarray,
    u: np.ndarray, v: np.ndarray, w: np.ndarray, th: np.ndarray,
    thl: np.ndarray, thv: np.ndarray, tk: np.ndarray, qt: np.ndarray,
    qv: np.ndarray, qc: np.ndarray, exner: np.ndarray, rstoch: np.ndarray,
    qc_bl: np.ndarray, cldfra_bl: np.ndarray, vt: np.ndarray,
    vq: np.ndarray, zw: np.ndarray,
    flt: np.float32, fltv: np.float32, flq: np.float32,
    pblh: np.float32, dx: np.float32, landsea: np.float32, ts: np.float32,
    psig_shcu: np.float32,
) -> dict:
    """One column of WRF ``DMP_mf`` (``module_bl_mynn.F:5679-6823``)."""

    nz = dz.size
    nup = DMP_NUP
    up_w = np.zeros((nz + 1, nup), dtype=np.float32)
    up_thl = np.zeros((nz + 1, nup), dtype=np.float32)
    up_thv = np.zeros((nz + 1, nup), dtype=np.float32)
    up_qt = np.zeros((nz + 1, nup), dtype=np.float32)
    up_qc = np.zeros((nz + 1, nup), dtype=np.float32)
    up_a = np.zeros((nz + 1, nup), dtype=np.float32)
    up_u = np.zeros((nz + 1, nup), dtype=np.float32)
    up_v = np.zeros((nz + 1, nup), dtype=np.float32)
    ent = np.full((nz, nup), F(0.001), dtype=np.float32)
    # W4 mixscalars exports: defaults for the no-plume path, where the
    # :6462-6496 limiter block never runs.
    up_a_prelimiter = up_a
    limiter_adjustment = F(1.0)
    edmf_a = np.zeros(nz, dtype=np.float32)
    edmf_w = np.zeros(nz, dtype=np.float32)
    edmf_qt = np.zeros(nz, dtype=np.float32)
    edmf_thl = np.zeros(nz, dtype=np.float32)
    edmf_ent = np.zeros(nz, dtype=np.float32)
    edmf_qc = np.zeros(nz, dtype=np.float32)
    s_aw = np.zeros(nz + 1, dtype=np.float32)
    s_awthl = np.zeros(nz + 1, dtype=np.float32)
    s_awqt = np.zeros(nz + 1, dtype=np.float32)
    s_awqv = np.zeros(nz + 1, dtype=np.float32)
    s_awqc = np.zeros(nz + 1, dtype=np.float32)
    s_awu = np.zeros(nz + 1, dtype=np.float32)
    s_awv = np.zeros(nz + 1, dtype=np.float32)
    qc_bl = qc_bl.astype(np.float32).copy()
    cldfra_bl = cldfra_bl.astype(np.float32).copy()
    vt = vt.astype(np.float32).copy()
    vq = vq.astype(np.float32).copy()
    nup2 = F(nup)

    # ---- module_bl_mynn.F:5939-5965 resolved-motion taper ----------------
    maxw = F(0.0)
    cloud_base = F(9000.0)
    # zw(kts) is zero, so the k=kts pass always runs and always sets k50;
    # the Fortran leaves it undefined only when zw(kts) > pblh+500.
    k50 = 1
    for k in range(nz - 1):
        if F(zw[k]) > F(pblh + F(500.0)):
            break
        wpbl = F(w[k])
        if F(w[k]) < F(0.0):
            wpbl = F(F(2.0) * F(w[k]))
        maxw = max(maxw, F(abs(wpbl)))
        if F(zw[k]) <= F(50.0):
            k50 = k + 1
        qc_sgs = max(F(qc[k]), F(qc_bl[k]))
        if qc_sgs > F(1.0e-5) and F(cldfra_bl[k]) >= F(0.5) \
                and cloud_base == F(9000.0):
            cloud_base = F(F(0.5) * F(zw[k] + zw[k + 1]))
    maxw = max(F(0.0), F(maxw - F(1.0)))
    psig_w = max(F(0.0), F(F(1.0) - maxw))
    psig_w = min(psig_w, F(psig_shcu))
    fltv2 = fltv
    if psig_w == F(0.0) and fltv > F(0.0):
        fltv2 = F(F(-1.0) * fltv)

    # ---- module_bl_mynn.F:5969-5992 superadiabatic surface layer ---------
    superadiabatic = False
    hux = F(-0.001) if F(landsea - F(1.5)) >= F(0.0) else F(-0.005)
    tvs = F(ts * F(F(1.0) + F(P608 * F(qv[0]))))
    for k in range(max(1, k50 - 1)):
        if k == 0:
            gradient = F(F(F(thv[0]) - tvs) / F(F(0.5) * F(dz[0])))
        else:
            gradient = F(F(F(thv[k]) - F(thv[k - 1]))
                         / F(F(0.5) * F(dz[k] + dz[k - 1])))
        if gradient < hux:
            superadiabatic = True
        else:
            superadiabatic = False
            break

    # ---- module_bl_mynn.F:6003-6035 plume-size criteria -------------------
    maxwidth = min(F(dx * DMP_DCUT), DMP_LMAX)
    maxwidth = min(maxwidth, F(F(1.1) * pblh))
    if F(landsea - F(1.5)) < F(0.0):
        maxwidth = min(maxwidth, F(F(0.5) * cloud_base))
    else:
        maxwidth = min(maxwidth, F(F(0.9) * cloud_base))
    wspd_pbl = F(np.sqrt(max(F(F(u[0] * u[0]) + F(v[0] * v[0])), F(0.01))))
    if F(landsea - F(1.5)) < F(0.0):
        width_flx = max(min(F(F(1000.0) * F(F(F(0.6) * _tanhf(
            F(F(fltv - F(0.040)) / F(0.04)))) + F(0.5))), F(1000.0)), F(0.0))
    else:
        width_flx = max(min(F(F(1000.0) * F(F(F(0.6) * _tanhf(
            F(F(fltv - F(0.007)) / F(0.02)))) + F(0.5))), F(1000.0)), F(0.0))
    maxwidth = min(maxwidth, width_flx)
    minwidth = DMP_LMIN
    if maxwidth >= F(DMP_LMAX - F(1.0)) and fltv > F(0.2):
        minwidth = F(DMP_LMIN + F(DMP_DLMIN * min(
            F(F(fltv - F(0.2)) / F(0.3)), F(1.0))))
    if maxwidth <= minwidth:
        nup2 = F(0.0)
        maxwidth = F(0.0)
    ktop = 0
    ztop = F(0.0)
    maxmf = F(0.0)

    rhoz = np.zeros(nz, dtype=np.float32)
    if fltv2 > F(0.002) and maxwidth > minwidth and superadiabatic:
        # ---- module_bl_mynn.F:6041-6076 number density -------------------
        # An2 (:6066, :6074) accumulates the plume areas for the commented-out
        # print at :6075 and is read nowhere else, so it is not carried here.
        cn = F(0.0)
        dl = F(F(maxwidth - minwidth) / F(nup - 1))
        for i in range(nup):
            length = F(minwidth + F(dl * F(i)))
            cn = F(cn + F(F(F(_powf(length, DMP_D) * F(length * length))
                             / F(dx * dx)) * dl))
        c_norm = F(DMP_ATOT / cn)
        acfac = F(F(F(0.5) * _tanhf(F(F(fltv2 - F(0.02)) / F(0.05))))
                  + F(0.5))
        if wspd_pbl <= F(10.0):
            ac_wsp = F(1.0)
        else:
            ac_wsp = F(F(1.0) - min(F(F(wspd_pbl - F(10.0)) / F(15.0)),
                                    F(1.0)))
        acfac = F(acfac * ac_wsp)
        for i in range(nup):
            length = F(minwidth + F(dl * F(i)))
            number = F(c_norm * _powf(length, DMP_D))
            up_a[0, i] = F(F(F(F(number * length) * length) / F(dx * dx))
                           * dl)
            up_a[0, i] = F(up_a[0, i] * acfac)

        # ---- module_bl_mynn.F:6079-6144 surface plume properties ---------
        z0 = F(50.0)
        pwmin = F(0.1)
        pwmax = F(0.4)
        wstar = max(F(1.0e-2),
                    _powf(F(F(GTR * fltv2) * pblh), ONETHIRD))
        qstar = F(max(flq, F(1.0e-5)) / wstar)
        thstar = F(flt / wstar)
        csigma = F(1.34)
        # env_subs is false, so exc_fac takes the land/water branch.
        exc_fac = F(F(0.58) * F(4.0)) if F(landsea - F(1.5)) >= F(0.0) \
            else F(0.58)
        exc_fac = F(exc_fac * ac_wsp)
        zratio = _powf(F(z0 / pblh), ONETHIRD)
        sigma_w = F(F(F(csigma * wstar) * zratio)
                    * F(F(1.0) - F(F(F(0.8) * z0) / pblh)))
        sigma_qt = F(F(csigma * qstar) * zratio)
        sigma_th = F(F(csigma * thstar) * zratio)
        wmin = min(F(sigma_w * pwmin), F(0.1))
        wmax = min(F(sigma_w * pwmax), F(0.5))
        for i in range(nup):
            up_w[0, i] = F(wmin + F(F(F(i + 1) / F(nup)) * F(wmax - wmin)))
            up_u[0, i] = _up(u[0], u[1], dz[0], dz[1])
            up_v[0, i] = _up(v[0], v[1], dz[0], dz[1])
            up_qc[0, i] = F(0.0)
            exc_heat = F(F(F(exc_fac * up_w[0, i]) * sigma_th) / sigma_w)
            up_thv[0, i] = F(_up(thv[0], thv[1], dz[0], dz[1]) + exc_heat)
            up_thl[0, i] = F(_up(thl[0], thl[1], dz[0], dz[1]) + exc_heat)
            exc_moist = F(F(F(exc_fac * up_w[0, i]) * sigma_qt) / sigma_w)
            up_qt[0, i] = F(_up(qt[0], qt[1], dz[0], dz[1]) + exc_moist)

        # rhoz and dxsa are module_bl_mynn.F:6161-6167.  The two blocks in
        # between are dead here: the mix_chem plume seeding at :6147-6153
        # (mix_chem false) and the envm_* initialisation at :6156-6160, whose
        # only readers are the det_* forms at :6595-6614 inside env_subs.
        for k in range(nz - 1):
            rhoz[k] = _up(rho[k], rho[k + 1], dz[k], dz[k + 1])
        rhoz[nz - 1] = rho[nz - 1]
        dxsa = F(F(1.0) - min(max(F(F(F(12000.0) - dx)
                                    / F(F(12000.0) - F(3000.0))), F(0.0)),
                              F(1.0)))

        # ---- module_bl_mynn.F:6170-6366 plume integration ----------------
        for i in range(nup):
            plume_qc = F(0.0)
            length = F(minwidth + F(dl * F(i)))
            for k in range(1, nz - 1):
                ent_wmin = F(F(0.3) + F(length * F(0.0005)))
                ent[k, i] = F(F(0.33) / F(
                    min(max(up_w[k - 1, i], ent_wmin), F(0.9)) * length))
                ent[k, i] = max(ent[k, i], F(0.0003))
                ramp = min(F(pblh + F(1500.0)), F(4000.0))
                if F(zw[k]) >= ramp:
                    ent[k, i] = F(ent[k, i]
                                  + F(F(F(zw[k]) - ramp) * F(5.0e-6)))
                ent[k, i] = F(ent[k, i] * F(F(1.0) - F(rstoch[k])))
                ent[k, i] = min(ent[k, i],
                                F(F(0.9) / F(zw[k + 1] - zw[k])))

                # pgfac is 0, so the pressure-gradient term contributes an
                # exact zero; it is written out so the expression tree is
                # WRF's rather than a rewrite.
                uk = _up(u[k], u[k + 1], dz[k], dz[k + 1])
                ukm1 = _up(u[k - 1], u[k], dz[k - 1], dz[k])
                vk = _up(v[k], v[k + 1], dz[k], dz[k + 1])
                vkm1 = _up(v[k - 1], v[k], dz[k - 1], dz[k])
                ent_exp = F(ent[k, i] * F(zw[k + 1] - zw[k]))
                ent_exm = F(ent_exp * F(0.3333))
                qtn = F(F(up_qt[k - 1, i] * F(F(1.0) - ent_exp))
                        + F(F(qt[k]) * ent_exp))
                thln = F(F(up_thl[k - 1, i] * F(F(1.0) - ent_exp))
                         + F(F(thl[k]) * ent_exp))
                un = F(F(F(up_u[k - 1, i] * F(F(1.0) - ent_exm))
                         + F(F(u[k]) * ent_exm))
                       + F(F(dxsa * F(0.0)) * F(uk - ukm1)))
                vn = F(F(F(up_v[k - 1, i] * F(F(1.0) - ent_exm))
                         + F(F(v[k]) * ent_exm))
                       + F(F(dxsa * F(0.0)) * F(vk - vkm1)))

                pk = _up(p[k], p[k + 1], dz[k], dz[k + 1])
                thvn, plume_qc = _condensation_edmf(
                    qtn, thln, pk, F(zw[k + 1]), plume_qc
                )
                thvk = _up(thv[k], thv[k + 1], dz[k], dz[k + 1])
                buoyancy = F(GRAV * F(F(thvn / thvk) - F(1.0)))
                bcoeff = F(0.15) if buoyancy > F(0.0) else F(0.2)
                previous = up_w[k - 1, i]
                step = min(F(zw[k] - zw[k - 1]), F(250.0))
                divisor = max(previous, F(0.2)) if previous < F(0.2) \
                    else previous
                wn = F(previous + F(F(
                    F(F(F(-2.0) * ent[k, i]) * previous)
                    + F(F(bcoeff * buoyancy) / divisor)) * step))
                limit = min(F(F(F(1.25) * F(zw[k] - zw[k - 1]))
                              / F(200.0)), F(2.0))
                if wn > F(previous + limit):
                    wn = F(previous + limit)
                if wn < F(previous - limit):
                    wn = F(previous - limit)
                wn = min(max(wn, F(0.0)), F(3.0))
                if k == 1 and wn == F(0.0):
                    nup2 = F(0.0)
                    break
                if wn > F(0.0):
                    up_w[k, i] = wn
                    up_thv[k, i] = thvn
                    up_thl[k, i] = thln
                    up_qt[k, i] = qtn
                    up_qc[k, i] = plume_qc
                    up_u[k, i] = un
                    up_v[k, i] = vn
                    up_a[k, i] = up_a[k - 1, i]
                    ktop = max(ktop, k + 1)
                else:
                    break
    else:
        nup2 = F(0.0)

    ktop = min(ktop, nz - 1)
    ztop = F(0.0) if ktop == 0 else F(zw[ktop - 1])

    if nup2 > F(0.0):
        # ---- module_bl_mynn.F:6404-6425 interface fluxes -----------------
        for i in range(nup):
            for k in range(nz - 1):
                s_aw[k + 1] = F(s_aw[k + 1] + F(
                    F(F(rhoz[k] * up_a[k, i]) * up_w[k, i]) * psig_w))
                s_awthl[k + 1] = F(s_awthl[k + 1] + F(F(F(
                    F(rhoz[k] * up_a[k, i]) * up_w[k, i]) * up_thl[k, i])
                    * psig_w))
                s_awqt[k + 1] = F(s_awqt[k + 1] + F(F(F(
                    F(rhoz[k] * up_a[k, i]) * up_w[k, i]) * up_qt[k, i])
                    * psig_w))
                s_awqc[k + 1] = F(s_awqc[k + 1] + F(F(F(
                    F(rhoz[k] * up_a[k, i]) * up_w[k, i]) * up_qc[k, i])
                    * psig_w))
                s_awqv[k + 1] = F(s_awqt[k + 1] - s_awqc[k + 1])
        # momentum_opt is 1.
        for i in range(nup):
            for k in range(nz - 1):
                s_awu[k + 1] = F(s_awu[k + 1] + F(F(F(
                    F(rhoz[k] * up_a[k, i]) * up_w[k, i]) * up_u[k, i])
                    * psig_w))
                s_awv[k + 1] = F(s_awv[k + 1] + F(F(F(
                    F(rhoz[k] * up_a[k, i]) * up_w[k, i]) * up_v[k, i])
                    * psig_w))

        # ---- module_bl_mynn.F:6462-6496 heat-flux limiter ----------------
        dzi = np.zeros(nz, dtype=np.float32)
        if s_aw[1] != F(0.0):
            dzi[0] = F(F(0.5) * F(dz[0] + dz[1]))
            flx1 = max(F(F(s_aw[1] * F(F(th[0]) - F(th[1]))) / dzi[0]),
                       F(1.0e-5))
        else:
            flx1 = F(0.0)
        adjustment = F(1.0)
        flt2 = max(flt, F(0.0))
        threshold = F(F(DMP_FLUXPORTION * flt2) / F(dz[0]))
        if flx1 > threshold and flx1 > F(0.0):
            adjustment = F(threshold / flx1)
            for array in (s_aw, s_awthl, s_awqt, s_awqc, s_awqv,
                          s_awu, s_awv):
                for k in range(nz + 1):
                    array[k] = F(array[k] * adjustment)
            # W4 mixscalars export: the :6485-6489 s_awqn* scalings happen
            # in the additive arm, which therefore needs the PRE-limiter
            # UPA (:6497 scales UPA only after every s_aw* line).
            up_a_prelimiter = up_a.copy()
            limiter_adjustment = adjustment
            for k in range(nz + 1):
                for i in range(nup):
                    up_a[k, i] = F(up_a[k, i] * adjustment)

        # ---- module_bl_mynn.F:6504-6524 plume means ----------------------
        for k in range(nz - 1):
            for i in range(nup):
                edmf_a[k] = F(edmf_a[k] + up_a[k, i])
                edmf_w[k] = F(edmf_w[k] + F(up_a[k, i] * up_w[k, i]))
                edmf_qt[k] = F(edmf_qt[k] + F(up_a[k, i] * up_qt[k, i]))
                edmf_thl[k] = F(edmf_thl[k] + F(up_a[k, i] * up_thl[k, i]))
                edmf_ent[k] = F(edmf_ent[k] + F(up_a[k, i] * ent[k, i]))
                edmf_qc[k] = F(edmf_qc[k] + F(up_a[k, i] * up_qc[k, i]))
        for k in range(nz - 1):
            if edmf_a[k] > F(0.0):
                edmf_w[k] = F(edmf_w[k] / edmf_a[k])
                edmf_qt[k] = F(edmf_qt[k] / edmf_a[k])
                edmf_thl[k] = F(edmf_thl[k] / edmf_a[k])
                edmf_ent[k] = F(edmf_ent[k] / edmf_a[k])
                edmf_qc[k] = F(edmf_qc[k] / edmf_a[k])
                edmf_a[k] = F(edmf_a[k] * psig_w)
                if F(edmf_a[k] * edmf_w[k]) > maxmf:
                    maxmf = F(edmf_a[k] * edmf_w[k])

        # ---- module_bl_mynn.F:6619-6625 interface exner and plume theta --
        edmf_th = np.zeros(nz, dtype=np.float32)
        for k in range(nz - 1):
            exneri = _up(exner[k], exner[k + 1], dz[k], dz[k + 1])
            edmf_th[k] = F(edmf_thl[k] + F(F(XLVCP / exneri) * edmf_qc[k]))
            dzi[k] = F(F(0.5) * F(dz[k] + dz[k + 1]))

        # ---- module_bl_mynn.F:6633-6764 shallow-cumulus cloud fraction ---
        for k in range(1, nz - 2):
            if k + 1 > ktop:
                break
            if not (F(F(0.5) * F(edmf_qc[k] + edmf_qc[k - 1])) > F(0.0)
                    and F(cldfra_bl[k]) < DMP_CF_THRESH):
                continue
            aup = _up(edmf_a[k], edmf_a[k - 1], dzi[k], dzi[k - 1])
            _thp = _up(edmf_th[k], edmf_th[k - 1], dzi[k], dzi[k - 1])
            qtp = _up(edmf_qt[k], edmf_qt[k - 1], dzi[k], dzi[k - 1])
            # esat/qsl are computed at :6641-6643 and never read again.
            if edmf_qc[k] > F(0.0) and edmf_qc[k - 1] > F(0.0):
                qcp = _up(edmf_qc[k], edmf_qc[k - 1], dzi[k], dzi[k - 1])
            else:
                qcp = max(edmf_qc[k], edmf_qc[k - 1])
            xl = mynn_xl_blend(F(tk[k]))
            qsat_tk = mynn_qsat_blend(F(tk[k]), F(p[k]))
            rsl = F(F(xl * qsat_tk) / F(RV * F(tk[k] * tk[k])))
            cpm = F(CP + F(F(qt[k]) * CPV))
            a_cb = F(F(1.0) / F(F(1.0) + F(F(xl * rsl) / cpm)))
            b9 = F(a_cb * rsl)
            q2p = F(XLVCP / F(exner[k]))
            pt = F(F(thl[k]) + F(F(q2p * qcp) * aup))
            bb = F(F(b9 * F(tk[k])) / pt)
            qww = F(F(1.0) + F(F(0.61) * F(qt[k])))
            alpha = F(F(0.61) * pt)
            beta = F(F(F(pt * xl) / F(F(tk[k]) * CP)) - F(F(1.61) * pt))
            sigq = F(F(F(10.0) * aup) * F(qtp - F(qt[k])))
            sigq = max(sigq, F(qsat_tk * F(0.02)))
            sigq = min(sigq, F(qsat_tk * F(0.25)))
            qmq = F(a_cb * F(F(qt[k]) - qsat_tk))
            q1 = F(qmq / sigq)
            mf_cf = min(max(F(F(0.5) + F(F(0.36) * _atanf(
                F(F(1.55) * q1)))), F(0.01)), F(0.6))
            if F(landsea - F(1.5)) >= F(0.0):
                mf_cf = max(mf_cf, F(F(1.2) * aup))
            else:
                mf_cf = max(mf_cf, F(F(1.8) * aup))
            mf_cf = min(mf_cf, F(F(5.0) * aup))
            if F(qcp * aup) > F(5.0e-5):
                qc_bl[k] = F(F(F(1.86) * F(qcp * aup)) - F(2.2e-5))
            else:
                qc_bl[k] = F(F(1.18) * F(qcp * aup))
            cldfra_bl[k] = mf_cf
            q1 = max(q1, F(-2.25))
            if q1 >= F(1.0):
                fng = F(1.0)
            elif q1 >= F(-1.7):
                fng = _expf(F(F(-0.4) * F(q1 - F(1.0))))
            elif q1 >= F(-2.5):
                fng = F(F(3.0) + _expf(F(F(-3.8) * F(q1 + F(1.7)))))
            else:
                fng = min(F(F(23.9) + _expf(F(F(-1.6) * F(q1 + F(2.5))))),
                          F(60.0))
            vt[k] = F(F(qww - F(F(F(F(F(1.5) * aup) * beta) * bb) * fng))
                      - F(1.0))
            vq[k] = F(F(alpha + F(F(F(F(F(1.5) * aup) * beta) * a_cb)
                                  * fng)) - TV0)

    # ---- module_bl_mynn.F:6771-6773 dry-plume sign convention ------------
    if ktop > 0:
        maxqc = F(edmf_qc[:ktop].max())
        if maxqc < F(1.0e-8):
            maxmf = F(F(-1.0) * maxmf)

    return {
        "edmf_a": edmf_a, "edmf_w": edmf_w, "edmf_qt": edmf_qt,
        "edmf_thl": edmf_thl, "edmf_ent": edmf_ent, "edmf_qc": edmf_qc,
        "qc_bl": qc_bl, "cldfra_bl": cldfra_bl, "vt": vt, "vq": vq,
        "s_aw": s_aw, "s_awthl": s_awthl, "s_awqt": s_awqt,
        "s_awqv": s_awqv, "s_awqc": s_awqc, "s_awu": s_awu,
        "s_awv": s_awv, "maxwidth": maxwidth, "ktop": ktop,
        "ztop": ztop, "maxmf": maxmf,
        # W4 mixscalars exports: the plume-edge terms the scalar_opt>0
        # accumulation (module_bl_mynn.F:6447-6456) consumes.  Additive
        # keys; woof.core.mynn_scalar_mix replays the qn plume lines
        # against these instead of recomputing plume dynamics.
        "up_w": up_w, "up_a": up_a_prelimiter, "ent": ent, "rhoz": rhoz,
        "psig_w": psig_w, "plume_active": bool(nup2 > F(0.0)),
        "limiter_adjustment": limiter_adjustment,
    }


def mynn_dmp_mf(
    values: Mapping[str, object],
    *,
    bl_mynn_edmf_mom: int = 1,
    bl_mynn_edmf_tke: int = 0,
    bl_mynn_mixscalars: int = 0,
    mix_chem: bool = False,
    spp_pbl: int = 0,
) -> dict[str, np.ndarray]:
    """Translate WRF ``DMP_mf`` (``module_bl_mynn.F:5679-6823``).

    The mass-flux plume scheme: eight plumes of increasing diameter are
    launched from the top of the first model layer and integrated upward with
    Tian and Kuang entrainment until their vertical velocity is driven to
    zero.  Their area-weighted fluxes are what ``mynn_tendencies`` needs in
    ``s_aw*``.

    Pinned identity: ``bl_mynn_edmf_mom=1``, ``bl_mynn_edmf_tke=0``,
    ``bl_mynn_mixscalars=0``, ``mix_chem=.false.`` and ``spp_pbl`` in {0,1}. Two
    compile-time parameters do most of the pruning: ``env_subs=.false.``
    (``module_bl_mynn.F:336``) skips the whole environmental subsidence and
    dynamic-detrainment block (``:6547-6617``), so ``sub_*`` and ``det_*``
    keep the zeros of ``:5926-5934``.  That block is also the only reader of
    the ``envm_*`` profiles (``:6595-6614``), which strands the plume-overshoot
    Froude limiter and the Asai-Kasahara detrainment rates at ``:6301-6339``:
    those statements still run in WRF but nothing they compute is ever read,
    so they are not transcribed.  ``bl_mynn_edmf_dd=0`` (``:330``) means no
    downdraft ever contributes.
    ``tke_opt=0`` and ``scalar_opt=0`` leave ``s_awqke`` and the five
    number-concentration fluxes zero.

    Returns the six ``edmf_*`` plume means, the seven live interface fluxes,
    the updated ``qc_bl``/``cldfra_bl``/``vt``/``vq`` columns, and the four
    diagnostics ``maxwidth``, ``ktop`` (WRF's one-based level), ``ztop`` and
    ``maxmf``.  The identically-zero output families are returned too so a
    caller cannot mistake absence for zero.
    """

    if bl_mynn_edmf_mom != 1 or type(bl_mynn_edmf_mom) is not int:
        raise ValueError("MYNN mass-flux lane requires bl_mynn_edmf_mom=1")
    if bl_mynn_edmf_tke != 0 or type(bl_mynn_edmf_tke) is not int:
        raise ValueError("MYNN mass-flux lane requires bl_mynn_edmf_tke=0")
    # W4 mixscalars admission (this wave; anchored fixtures
    # w4-oracle-fixtures): scalar_opt=1 makes the s_awqn*
    # accumulation at module_bl_mynn.F:6447-6456 live, computed by the
    # additive woof.core.mynn_scalar_mix arm from this transcription's
    # own plume-edge terms.  Other nonzero values stay refused.
    if bl_mynn_mixscalars not in (0, 1) or \
            type(bl_mynn_mixscalars) is not int:
        raise ValueError(
            "MYNN mass-flux lane requires bl_mynn_mixscalars in {0,1}"
        )
    if mix_chem is not False:
        raise ValueError("MYNN mass-flux lane requires mix_chem false")
    if type(spp_pbl) is not int or spp_pbl not in (0, 1):
        raise ValueError("MYNN mass-flux requires spp_pbl in {0,1}")
    missing = [name for name in MYNN_DMP_MF_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN mass-flux inputs: {', '.join(missing)}")

    columns = {
        name: np.asarray(values[name], dtype=np.float32)
        for name in MYNN_DMP_MF_COLUMN_INPUTS
    }
    shapes = {array.shape for array in columns.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 2:
        raise ValueError("MYNN mass-flux columns must share shape (ncol,nz)")
    ncol, nz = next(iter(shapes))
    if nz < 5:
        raise ValueError("MYNN mass flux requires nz >= 5")
    interface = np.asarray(values["zw"], dtype=np.float32)
    if interface.shape != (ncol, nz + 1):
        raise ValueError("MYNN mass-flux zw must have shape (ncol,nz+1)")
    scalars: dict[str, np.ndarray] = {}
    for name in MYNN_DMP_MF_SCALAR_INPUTS:
        try:
            scalars[name] = np.broadcast_to(
                np.asarray(values[name], dtype=np.float32), (ncol,)
            )
        except ValueError as exc:
            raise ValueError(f"{name} is not broadcastable to ncol") from exc
    if any(not np.isfinite(array).all()
           for array in (*columns.values(), interface, *scalars.values())):
        raise ValueError("MYNN mass-flux inputs must be finite")
    if np.any(columns["dz"] <= 0.0):
        raise ValueError("MYNN mass-flux layer depths must be positive")
    if np.any(columns["p"] <= 0.0) or np.any(columns["rho"] <= 0.0):
        raise ValueError("MYNN mass-flux p and rho must be positive")
    if np.any(columns["exner"] <= 0.0) or np.any(columns["tk"] <= 0.0):
        raise ValueError("MYNN mass-flux exner and tk must be positive")
    if np.any(scalars["pblh"] <= 0.0) or np.any(scalars["dx"] <= 0.0):
        raise ValueError("MYNN mass-flux pblh and dx must be positive")

    outputs: dict[str, np.ndarray] = {
        name: np.zeros((ncol, nz), dtype=np.float32)
        for name in (*MYNN_DMP_MF_LAYER_OUTPUTS, *MYNN_DMP_MF_ZERO_OUTPUTS)
    }
    outputs.update({
        name: np.zeros((ncol, nz + 1), dtype=np.float32)
        for name in (*MYNN_DMP_MF_INTERFACE_OUTPUTS,
                     *MYNN_DMP_MF_ZERO_INTERFACE_OUTPUTS)
    })
    outputs["maxwidth"] = np.zeros(ncol, dtype=np.float32)
    outputs["ztop"] = np.zeros(ncol, dtype=np.float32)
    outputs["maxmf"] = np.zeros(ncol, dtype=np.float32)
    outputs["ktop"] = np.zeros(ncol, dtype=np.int32)
    qn_columns: dict[str, np.ndarray] = {}
    if bl_mynn_mixscalars == 1:
        missing_qn = [
            name for name in MYNN_DMP_MF_QN_COLUMN_INPUTS
            if name not in values
        ]
        if missing_qn:
            raise TypeError(
                "missing MYNN mixscalars mass-flux inputs: "
                + ", ".join(missing_qn)
            )
        for name in MYNN_DMP_MF_QN_COLUMN_INPUTS:
            array = np.asarray(values[name], dtype=np.float32)
            if array.shape != (ncol, nz):
                raise ValueError(f"{name} must have shape (ncol,nz)")
            qn_columns[name] = array
    for column in range(ncol):
        result = _dmp_mf_column(
            *(columns[name][column] for name in MYNN_DMP_MF_COLUMN_INPUTS),
            interface[column],
            *(F(scalars[name][column]) for name in MYNN_DMP_MF_SCALAR_INPUTS),
        )
        for name in (*MYNN_DMP_MF_LAYER_OUTPUTS,
                     *MYNN_DMP_MF_INTERFACE_OUTPUTS):
            outputs[name][column] = result[name]
        for name in MYNN_DMP_MF_SCALAR_OUTPUTS:
            outputs[name][column] = result[name]
        if bl_mynn_mixscalars == 1:
            # W4 mixscalars: the scalar_opt>0 accumulation (:6447-6456),
            # replayed by the additive arm from this column's own
            # plume-edge exports.  s_awqke stays a structural zero
            # (tke_opt=0 pin unchanged).
            from woof.core.mynn_scalar_mix import dmp_qn_flux_column
            for name in MYNN_DMP_MF_QN_COLUMN_INPUTS:
                outputs[f"s_aw{name}"][column] = dmp_qn_flux_column(
                    qn_columns[name][column], columns["dz"][column],
                    interface[column], result["up_w"], result["up_a"],
                    result["ent"], result["rhoz"], result["psig_w"],
                    result["plume_active"], result["limiter_adjustment"],
                )
    return outputs


# ===========================================================================
# module_bl_mynn.F:7525-7623 phim / phih, and :360-1453 mynn_bl_driver.
# ===========================================================================

# module_bl_mynn.F:7537-7539 and :7590-7592.
PHI_AM_ST = F(6.1)
PHI_BM_ST = F(2.5)
PHI_RBM_ST = F(F(1.0) / PHI_BM_ST)
PHI_AH_ST = F(5.3)
PHI_BH_ST = F(1.1)
PHI_RBH_ST = F(F(1.0) / PHI_BH_ST)
PHI_AM_UNST = F(10.0)
PHI_AH_UNST = F(34.0)
# module_bl_mynn.F:272-273.
CPHM_UNST = F(16.0)
CPHH_UNST = F(16.0)


def _phi_stable(zet, a_st, b_st, rb_st) -> np.float32:
    """The ``zet >= 0`` arm shared by ``phim`` and ``phih``.

    Cheng and Brutsaert (2005).  Only ``powf`` appears here and the port
    routes it onto the verified glibc transcription, so this arm is bitwise.
    """

    dummy_0 = F(F(1.0) + _powf(zet, b_st))
    dummy_1 = F(zet + _powf(dummy_0, rb_st))
    dummy_11 = F(F(1.0) + F(_powf(dummy_0, F(rb_st - F(1.0)))
                            * _powf(zet, F(b_st - F(1.0)))))
    dummy_2 = F(F(-a_st / dummy_1) * dummy_11)
    return F(F(1.0) - F(zet * dummy_2))


def _phi_unstable(zet, phi, dummy_psi, a_unst) -> np.float32:
    """The ``zet < 0`` tail shared by ``phim`` and ``phih`` (Grachev 2000)."""

    dummy_0 = F(F(1.0) - F(a_unst * zet))
    dummy_1 = _powf(dummy_0, F(0.333333))
    dummy_11 = F(F(F(-0.33333) * a_unst)
                 * _powf(dummy_0, F(-0.6666667)))
    dummy_2 = F(F(0.33333)
                * F(F(_powf(dummy_1, F(2.0)) + dummy_1) + F(1.0)))
    dummy_22 = F(F(F(0.3333) * dummy_11) * F(F(F(2.0) * dummy_1) + F(1.0)))
    dummy_3 = F(F(0.57735) * F(F(F(2.0) * dummy_1) + F(1.0)))
    dummy_33 = F(F(1.1547) * dummy_11)
    dummy_4 = F(F(F(F(1.5) * _glibc_logf(dummy_2))
                  - F(F(1.73205) * _atanf(dummy_3))) + F(1.813799364))
    dummy_44 = F(F(F(F(1.5) / dummy_2) * dummy_22)
                 - F(F(F(1.73205) * dummy_33)
                     / F(F(1.0) + F(dummy_3 * dummy_3))))
    dummy_0 = F(zet * zet)
    dummy_1 = F(F(1.0) / F(F(1.0) + dummy_0))
    dummy_11 = F(F(2.0) * zet)
    dummy_2 = F(F(F(F(F(1.0) - phi) / zet) + F(dummy_11 * dummy_4))
                + F(dummy_0 * dummy_44))
    dummy_2 = F(dummy_2 * dummy_1)
    dummy_22 = F(F(-F(dummy_11 * F(dummy_psi + F(dummy_0 * dummy_4))))
                 * F(dummy_1 * dummy_1))
    return F(F(1.0) - F(zet * F(dummy_2 + dummy_22)))


def mynn_phim(zet: object) -> np.float32:
    """Translate WRF ``phim`` (``module_bl_mynn.F:7525-7574``).

    ``bl_mynn_stfunc`` is a compile-time ``1`` at ``:340``, so this -- not the
    Kansas form at ``:1085-1091`` -- is what builds ``pmz`` for
    ``mym_predict``.  The Fortran returns ``phi_m``, not ``phi_m - zet``; the
    subtraction happens at the call site (``:1095``).

    Both arms are bitwise over an 814-point sweep of the driver's clamped z/L
    range.  The unstable arm was not while :func:`_atanf` rounded an FP64
    evaluation: glibc's ``atanf`` is faithfully rounded, not correctly
    rounded, and the ``(1 - phi_m)/zet`` cancellation that follows amplifies a
    one-ULP disagreement into tens.  22 of 814 ``phim`` values and 9 of 814
    ``phih`` values differed, worst case 80 and 84 ULP, every one of them with
    ``zet < 0``.  All four counts are zero on the verified shim.
    """

    zet = F(zet)
    if zet >= F(0.0):
        return _phi_stable(zet, PHI_AM_ST, PHI_BM_ST, PHI_RBM_ST)
    dummy_0 = _powf(F(F(1.0) - F(CPHM_UNST * zet)), F(0.25))
    phi_m = F(F(1.0) / dummy_0)
    dummy_psi = F(
        F(F(F(F(2.0) * _glibc_logf(F(F(0.5) * F(F(1.0) + dummy_0))))
            + _glibc_logf(F(F(0.5) * F(F(1.0) + F(dummy_0 * dummy_0)))))
          - F(F(2.0) * _atanf(dummy_0))) + F(1.570796)
    )
    return _phi_unstable(zet, phi_m, dummy_psi, PHI_AM_UNST)


def mynn_phih(zet: object) -> np.float32:
    """Translate WRF ``phih`` (``module_bl_mynn.F:7578-7623``).

    See :func:`mynn_phim` for the arm-by-arm accuracy statement.
    """

    zet = F(zet)
    if zet >= F(0.0):
        return _phi_stable(zet, PHI_AH_ST, PHI_BH_ST, PHI_RBH_ST)
    dummy_0 = _powf(F(F(1.0) - F(CPHH_UNST * zet)), F(0.5))
    phh = F(F(1.0) / dummy_0)
    dummy_psi = F(F(2.0) * _glibc_logf(F(F(0.5) * F(F(1.0) + dummy_0))))
    return _phi_unstable(zet, phh, dummy_psi, PHI_AH_UNST)


#: What the driver needs from the host, per column and per level.
MYNN_DRIVER_LAYER_INPUTS = (
    "dz", "u", "v", "w", "th", "sqv", "sqc", "sqi", "sqs", "p", "exner",
    "rho", "tk",
)
#: One value per column.
MYNN_DRIVER_SCALAR_INPUTS = (
    "dx", "xland", "ts", "ps", "ust", "hfx", "qfx", "wspd", "uoce", "voce",
)
#: Carried across steps.  ``sh``/``sm`` are in this list because WRF declares
#: Sh3D/Sm3D ``intent(out)`` at :511 and then reads them back at :952 on every
#: step after the first; with gfortran and an explicit-shape actual argument
#: that is a pass-by-reference no-op, so the previous step's values survive.
MYNN_DRIVER_STATE = (
    "qke", "tsq", "qsq", "cov", "el", "sh", "sm",
    "qc_bl", "qi_bl", "cldfra_bl",
)
MYNN_DRIVER_INPUTS = (
    *MYNN_DRIVER_LAYER_INPUTS, *MYNN_DRIVER_SCALAR_INPUTS,
    *MYNN_DRIVER_STATE, "pblh", "kpbl", "rmol",
)


def _driver_zw(dz: np.ndarray, nz: int) -> np.ndarray:
    """``zw`` from ``dz``, the way the driver builds it at :1001-1017."""

    ncol = dz.shape[0]
    zw = np.zeros((ncol, nz + 1), dtype=np.float32)
    for column in range(ncol):
        for k in range(1, nz + 1):
            zw[column, k] = F(zw[column, k - 1] + dz[column, k - 1])
    return zw


def mynn_bl_driver(
    values: Mapping[str, object],
    *,
    initflag: int,
    delt: object,
    restart: bool = False,
    cycling: bool = False,
    closure: float = 2.6,
    bl_mynn_cloudpdf: int = 2,
    bl_mynn_mixlength: int = 1,
    bl_mynn_edmf: int = 1,
    bl_mynn_edmf_mom: int = 1,
    bl_mynn_edmf_tke: int = 0,
    bl_mynn_mixscalars: int = 0,
    bl_mynn_cloudmix: int = 1,
    bl_mynn_mixqt: int = 0,
    bl_mynn_output: int = 0,
    bl_mynn_tkeadvect: bool = False,
    icloud_bl: int = 1,
    tke_budget: int = 0,
    spp_pbl: int = 0,
    mix_chem: bool = False,
    flag_qc: bool = True,
    flag_qi: bool = True,
    flag_qs: bool = False,
    flag_qnc: bool = False,
    flag_qni: bool = False,
    flag_qnwfa: bool = False,
    flag_qnifa: bool = False,
    flag_qnbca: bool = False,
    flag_ozone: bool = False,
) -> dict[str, np.ndarray]:
    """Translate WRF ``mynn_bl_driver`` (``module_bl_mynn.F:360-1453``).

    This is the assembly, not a leaf: column extraction, the surface-flux
    construction, the call order, and the write-back.  Every routine it calls
    is one of the pinned transcriptions in this module.

    Admitted identity, matching the WRF registry defaults for
    ``bl_pbl_physics=5``: ``bl_mynn_cloudpdf=2``, ``bl_mynn_mixlength=1 or 2``,
    ``bl_mynn_edmf=1``, ``bl_mynn_edmf_mom=1``, ``bl_mynn_edmf_tke=0``,
    ``bl_mynn_mixscalars=0``, ``bl_mynn_output=0``, ``bl_mynn_cloudmix=1``,
    ``bl_mynn_mixqt=0``, ``icloud_bl=1``, ``closure=2.6``,
    ``bl_mynn_tkeadvect`` logical, ``tke_budget=0``, ``spp_pbl`` in {0,1},
    ``mix_chem`` false, ``restart``/``cycling`` false, ``FLAG_QC``/``FLAG_QI``
    true, ``FLAG_QS`` either Registry-derived boolean value, and every other
    species flag false.  Three module parameters do the
    rest of the pruning: ``bl_mynn_topdown=0`` (``:328``) kills
    ``topdown_cloudrad`` and pins ``TKEprodTD`` to zero,
    ``bl_mynn_edmf_dd=0`` (``:330``) kills ``DDMF_JPL`` and pins every
    ``sd_aw*`` to zero, and ``bl_mynn_stfunc=1`` (``:340``) selects
    :func:`mynn_phim` / :func:`mynn_phih` over the Kansas forms.

    ``initflag > 0`` runs the cold start at ``:658-857``: the state arrays are
    zeroed, ``qke`` is seeded from the Koracin and Berkowicz taper at ``:775``,
    and ``mym_initialize`` fills ``el``, ``tsq``, ``qsq``, ``cov``, ``sm`` and
    ``sh``.  ``initflag == 0`` consumes the state a previous call produced.

    Not bitwise end to end.  ``pmz``/``phh`` come from ``phim``/``phih``, whose
    unstable arm inherits glibc's non-correctly-rounded ``atanf`` (see
    :func:`mynn_phim`); they reach only ``mym_predict``'s surface boundary
    condition, so the residue is confined to ``qke``, ``tsq``, ``qsq``,
    ``cov`` on columns with ``zet < 0`` and to whatever those carry forward.
    Everything that runs before ``mym_predict`` -- the assembly, ``get_pblh``,
    ``scale_aware``, ``mym_initialize``, ``mym_condensation``, ``DMP_mf`` and
    ``mym_turbulence`` -- is bitwise, and so are the tendencies on the first
    step, where ``mym_predict`` has not yet fed anything back.
    """

    if type(initflag) is not int:
        raise TypeError("MYNN driver initflag must be an int")
    if restart is not False or cycling is not False:
        raise ValueError("MYNN driver lane requires restart and cycling false")
    if bl_mynn_edmf != 1 or type(bl_mynn_edmf) is not int:
        raise ValueError("MYNN driver lane requires bl_mynn_edmf=1")
    if bl_mynn_output != 0 or type(bl_mynn_output) is not int:
        raise ValueError("MYNN driver lane requires bl_mynn_output=0")
    if type(bl_mynn_tkeadvect) is not bool:
        raise TypeError("bl_mynn_tkeadvect must be a bool")
    if icloud_bl != 1 or type(icloud_bl) is not int:
        raise ValueError("MYNN driver lane requires icloud_bl=1")
    if tke_budget != 0 or type(tke_budget) is not int:
        raise ValueError("MYNN driver lane requires tke_budget=0")
    if type(spp_pbl) is not int or spp_pbl not in (0, 1):
        raise ValueError("MYNN driver requires spp_pbl in {0,1}")
    if mix_chem is not False:
        raise ValueError("MYNN driver lane requires mix_chem false")
    # W4 full admission (mf-close2, Stage B): the driver now FEEDS the
    # mixscalars arms its leaf routines already implement -- qn columns
    # into DMP_mf's scalar_opt>0 accumulation and the five s_awqn*
    # interfaces into the tendency solve (module_bl_mynn.F binds them at
    # the same call sites as the admitted s_aw* set).  Any value outside
    # {0,1} has no meaning: WRF defines the key at 0 or 1.
    if bl_mynn_mixscalars not in (0, 1) or \
            type(bl_mynn_mixscalars) is not int:
        raise ValueError(
            "MYNN driver lane requires bl_mynn_mixscalars in {0,1}"
        )
    _tendency_flag_identity(
        flag_qc, flag_qi, flag_qs, flag_qnc, flag_qni,
        flag_qnwfa, flag_qnifa, flag_qnbca, flag_ozone,
        bl_mynn_mixscalars,
    )
    missing = [name for name in MYNN_DRIVER_INPUTS if name not in values]
    if missing:
        raise TypeError(f"missing MYNN driver inputs: {', '.join(missing)}")

    layers = {
        name: np.array(values[name], dtype=np.float32, copy=True)
        for name in (*MYNN_DRIVER_LAYER_INPUTS, *MYNN_DRIVER_STATE)
    }
    shapes = {array.shape for array in layers.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("MYNN driver columns must share shape (ncol,nz)")
    ncol, nz = next(iter(shapes))
    if nz < 4:
        raise ValueError("MYNN driver requires nz >= 4")
    # Stage B: the qn columns ride into the driver only under the key --
    # the mixscalars=0 assembly reads none of these names, so the admitted
    # trajectory cannot move.  No unit conversion: the WRF wrapper passes
    # number concentrations straight through (module_bl_mynn_wrapper.F
    # converts moisture species only).
    qn_layers: dict[str, np.ndarray] = {}
    if bl_mynn_mixscalars == 1:
        missing_qn = [name for name in MYNN_TENDENCIES_QN_LAYER_INPUTS
                      if name not in values]
        if missing_qn:
            raise TypeError("missing MYNN driver mixscalars inputs: "
                            + ", ".join(missing_qn))
        for name in MYNN_TENDENCIES_QN_LAYER_INPUTS:
            qn = np.array(values[name], dtype=np.float32, copy=True)
            if qn.shape != (ncol, nz):
                raise ValueError(
                    f"{name} must have shape ({ncol},{nz}), got {qn.shape}")
            qn_layers[name] = qn
    scalars = {
        name: np.broadcast_to(
            np.asarray(values[name], dtype=np.float32), (ncol,)
        ).astype(np.float32)
        for name in (*MYNN_DRIVER_SCALAR_INPUTS, "pblh", "rmol")
    }
    kpbl = np.asarray(values["kpbl"], dtype=np.int32).reshape(ncol).copy()
    delt = F(delt)
    if not np.isfinite(delt) or delt <= 0.0:
        raise ValueError("MYNN driver delt must be positive and finite")

    zero_column = np.zeros((ncol, nz), dtype=np.float32)
    if spp_pbl:
        if "rstoch" not in values:
            raise TypeError("MYNN stochastic driver requires rstoch[ncol,nz]")
        rstoch = np.asarray(values["rstoch"], dtype=np.float32)
        if rstoch.shape != (ncol, nz) or not np.isfinite(rstoch).all():
            raise ValueError("MYNN stochastic driver rstoch must be finite with shape (ncol,nz)")
    else:
        rstoch = zero_column
    zero_interface = np.zeros((ncol, nz + 1), dtype=np.float32)
    zw = _driver_zw(layers["dz"], nz)
    # module_bl_mynn.F:1240-1242: the driver replaces both qs and sqs with a
    # zero column in the tendency solve.  Condensation separately receives
    # the real sqs below when FLAG_QS is true.
    kzero = zero_column.copy()

    if initflag > 0:
        # module_bl_mynn.F:674-688 cold start.  qi_bl is conspicuously absent
        # from the zeroing list at :681-682; it is left alone here for the
        # same reason, and nothing reads it before mym_condensation
        # overwrites it.
        for name in ("sh", "sm", "el", "tsq", "qsq", "cov",
                     "cldfra_bl", "qc_bl", "qke"):
            layers[name][...] = 0.0
        thl_init = np.empty((ncol, nz), dtype=np.float32)
        thetav_init = np.empty((ncol, nz), dtype=np.float32)
        qke_seed = np.empty((ncol, nz), dtype=np.float32)
        for column in range(ncol):
            ust = F(scalars["ust"][column])
            for k in range(nz):
                exner = F(layers["exner"][column, k])
                sqc = F(layers["sqc"][column, k])
                sqi = F(layers["sqi"][column, k])
                sqv = F(layers["sqv"][column, k])
                th = F(layers["th"][column, k])
                thl_init[column, k] = F(F(th - F(F(XLVCP / exner) * sqc))
                                        - F(F(XLSCP / exner) * sqi))
                thetav_init[column, k] = F(th * F(F(1.0) + F(P608 * sqv)))
                qke_seed[column, k] = F(
                    F(F(5.0) * ust)
                    * max(F(F(F(ust * F(700.0)) - zw[column, k])
                             / F(max(ust, F(0.01)) * F(700.0))), F(0.01))
                )
        pblh_init, kpbl_init = mynn_get_pblh(
            thetav_init, qke_seed, zw, layers["dz"], scalars["xland"]
        )
        scalars["pblh"] = np.asarray(pblh_init, dtype=np.float32)
        kpbl = np.asarray(kpbl_init, dtype=np.int32)
        psig_bl_init, _ = mynn_scale_aware(scalars["dx"], scalars["pblh"])
        seeded = mynn_initialize_default(
            {
                "dz": layers["dz"], "u": layers["u"], "v": layers["v"],
                # WRF's cold call passes sqv, not total water sqw. Passing
                # condensate here changes the initial cloud-column TKE.
                "thl": thl_init, "qw": layers["sqv"], "theta": layers["th"],
                "thetav": thetav_init, "cldfra": layers["cldfra_bl"],
                "edmf_w": zero_column, "edmf_a": zero_column,
                "sm": layers["sm"], "sh": layers["sh"], "qke": qke_seed,
                "zw": zw, "xland": scalars["xland"], "dx": scalars["dx"],
                "rmo": scalars["rmol"], "ust": scalars["ust"],
                "zi": scalars["pblh"], "psig_bl": psig_bl_init,
            },
            initialize_qke=True,
            bl_mynn_mixlength=bl_mynn_mixlength,
            spp_pbl=spp_pbl,
        )
        for name in ("el", "qke", "tsq", "qsq", "cov", "sm", "sh"):
            layers[name][...] = seeded[name]

    # WRF :838 initializes the carrier on the cold call. On warm calls
    # :862 imports its transported value before any regular MYNN solve.
    if bl_mynn_tkeadvect:
        if "qke_adv" not in values:
            raise TypeError("MYNN TKE advection requires qke_adv[ncol,nz]")
        carrier = np.asarray(values["qke_adv"], dtype=np.float32)
        if carrier.shape != (ncol, nz):
            raise ValueError("qke_adv must share the MYNN column shape")
        if initflag <= 0:
            layers["qke"][...] = carrier

    # ---- module_bl_mynn.F:866-1017 per-column assembly -------------------
    thl = np.empty((ncol, nz), dtype=np.float32)
    sqw = np.empty((ncol, nz), dtype=np.float32)
    thetav = np.empty((ncol, nz), dtype=np.float32)
    qv1 = np.empty((ncol, nz), dtype=np.float32)
    for column in range(ncol):
        for k in range(nz):
            exner = F(layers["exner"][column, k])
            sqc = F(layers["sqc"][column, k])
            sqi = F(layers["sqi"][column, k])
            sqv = F(layers["sqv"][column, k])
            th = F(layers["th"][column, k])
            qv1[column, k] = F(sqv / F(F(1.0) - sqv))
            sqw[column, k] = F(F(sqv + sqc) + sqi)
            thl[column, k] = F(F(th - F(F(XLVCP / exner) * sqc))
                               - F(F(XLSCP / exner) * sqi))
            thetav[column, k] = F(th * F(F(1.0) + F(P608 * sqv)))

    pblh_new, kpbl_new = mynn_get_pblh(
        thetav, layers["qke"], zw, layers["dz"], scalars["xland"]
    )
    scalars["pblh"] = np.asarray(pblh_new, dtype=np.float32)
    kpbl = np.asarray(kpbl_new, dtype=np.int32)
    psig_bl, psig_shcu = mynn_scale_aware(scalars["dx"], scalars["pblh"])

    # ---- module_bl_mynn.F:1057-1097 surface fluxes and z/L ---------------
    flt = np.empty(ncol, dtype=np.float32)
    fltv = np.empty(ncol, dtype=np.float32)
    flq = np.empty(ncol, dtype=np.float32)
    flqv = np.empty(ncol, dtype=np.float32)
    flqc = np.zeros(ncol, dtype=np.float32)
    th_sfc = np.empty(ncol, dtype=np.float32)
    pmz = np.empty(ncol, dtype=np.float32)
    phh = np.empty(ncol, dtype=np.float32)
    rmol = np.empty(ncol, dtype=np.float32)
    for column in range(ncol):
        rho0 = F(layers["rho"][column, 0])
        exner0 = F(layers["exner"][column, 0])
        ust = F(scalars["ust"][column])
        cpm = F(CP * F(F(1.0) + F(F(0.84) * qv1[column, 0])))
        flqv[column] = F(F(scalars["qfx"][column]) / rho0)
        th_sfc[column] = F(F(scalars["ts"][column]) / exner0)
        flq[column] = F(flqv[column] + flqc[column])
        flt[column] = F(F(F(scalars["hfx"][column]) / F(rho0 * cpm))
                        - F(F(XLVCP * flqc[column]) / exner0))
        fltv[column] = F(flt[column] + F(F(flqv[column] * P608)
                                         * th_sfc[column]))
        ust3 = F(F(ust * ust) * ust)
        rmol[column] = F(-F(F(F(KARMAN * GTR) * fltv[column])
                            / max(ust3, F(1.0e-6))))
        zet = F(F(F(0.5) * F(layers["dz"][column, 0])) * rmol[column])
        zet = max(zet, F(-20.0))
        zet = min(zet, F(20.0))
        pmz[column] = F(mynn_phim(zet) - zet)
        phh[column] = mynn_phih(zet)
    scalars["rmol"] = rmol

    # ---- module_bl_mynn.F:1104-1112 subgrid condensation -----------------
    # WRF :1104-1106 selects the real snow column under FLAG_QS and kzero
    # otherwise.  This is snow's live MYNN v4.6.1 consumer.
    condensed = mynn_condensation_default(
        {
            "dz": layers["dz"], "zw": zw, "th": layers["th"], "thl": thl,
            "qw": sqw, "qv": layers["sqv"], "qc": layers["sqc"],
            "qi": layers["sqi"],
            "qs": layers["sqs"] if flag_qs else kzero, "p": layers["p"],
            "exner": layers["exner"], "tsq": layers["tsq"],
            "qsq": layers["qsq"], "cov": layers["cov"], "sh": layers["sh"],
            "el": layers["el"], "rstoch": rstoch,
            "vt": zero_column, "vq": zero_column, "sgm": zero_column,
            "xland": scalars["xland"], "dx": scalars["dx"],
            "pblh": scalars["pblh"], "hfx": scalars["hfx"], "rmo": rmol,
        },
        bl_mynn_cloudpdf=bl_mynn_cloudpdf,
        spp_pbl=spp_pbl,
    )
    qc_bl = np.array(condensed["qc_bl"], dtype=np.float32, copy=True)
    qi_bl = np.array(condensed["qi_bl"], dtype=np.float32, copy=True)
    cldfra_bl = np.array(condensed["cldfra"], dtype=np.float32, copy=True)
    vt = np.array(condensed["vt"], dtype=np.float32, copy=True)
    vq = np.array(condensed["vq"], dtype=np.float32, copy=True)
    sgm = np.array(condensed["sgm"], dtype=np.float32, copy=True)

    # ---- module_bl_mynn.F:1131-1169 mass flux ----------------------------
    plumes = mynn_dmp_mf(
        {
            "dz": layers["dz"], "zw": zw, "p": layers["p"],
            "rho": layers["rho"], "u": layers["u"], "v": layers["v"],
            "w": layers["w"], "th": layers["th"], "thl": thl,
            "thv": thetav, "tk": layers["tk"], "qt": sqw,
            "qv": layers["sqv"], "qc": layers["sqc"],
            "exner": layers["exner"], "rstoch": rstoch,
            "qc_bl": qc_bl, "cldfra_bl": cldfra_bl, "vt": vt, "vq": vq,
            "sgm": sgm,
            "flt": flt, "fltv": fltv, "flq": flq,
            "pblh": scalars["pblh"], "dx": scalars["dx"],
            "landsea": scalars["xland"], "ts": th_sfc,
            "psig_shcu": psig_shcu,
            # Stage B: qn columns feed the scalar_opt>0 accumulation; an
            # empty dict under mixscalars=0 adds no key at all.
            **qn_layers,
        },
        bl_mynn_edmf_mom=bl_mynn_edmf_mom,
        bl_mynn_edmf_tke=bl_mynn_edmf_tke,
        bl_mynn_mixscalars=bl_mynn_mixscalars,
        mix_chem=mix_chem,
        spp_pbl=spp_pbl,
    )
    qc_bl = plumes["qc_bl"]
    cldfra_bl = plumes["cldfra_bl"]
    vt = plumes["vt"]
    vq = plumes["vq"]

    # ---- module_bl_mynn.F:1192-1210 diffusivities ------------------------
    turbulence = mynn_turbulence_default(
        {
            "dz": layers["dz"], "zw": zw, "u": layers["u"],
            "v": layers["v"], "thl": thl, "thetav": thetav,
            "ql": layers["sqc"], "qw": sqw, "qke": layers["qke"],
            "tsq": layers["tsq"], "qsq": layers["qsq"],
            "cov": layers["cov"], "vt": vt, "vq": vq,
            "theta": layers["th"], "cldfra": cldfra_bl,
            "edmf_w": plumes["edmf_w"], "edmf_a": plumes["edmf_a"],
            "tkeprodtd": zero_column, "xland": scalars["xland"],
            "dx": scalars["dx"], "rmo": rmol, "flt": flt, "fltv": fltv,
            "flq": flq, "zi": scalars["pblh"], "psig_bl": psig_bl,
            "psig_shcu": psig_shcu, "rstoch": rstoch,
        },
        closure=closure,
        bl_mynn_mixlength=bl_mynn_mixlength,
        spp_pbl=spp_pbl,
    )

    # ---- module_bl_mynn.F:1215-1221 prognostic solve ---------------------
    predicted = mynn_predict_default(
        {
            "dz": layers["dz"], "rho": layers["rho"],
            "dfq": turbulence["dfq"], "pdk": turbulence["pdk"],
            "pdt": turbulence["pdt"], "pdq": turbulence["pdq"],
            "pdc": turbulence["pdc"], "el": turbulence["el"],
            "s_aw": plumes["s_aw"], "s_awqke": zero_interface,
            "ust": scalars["ust"], "flt": flt, "flq": flq,
            "pmz": pmz, "phh": phh,
            "delt": np.full(ncol, delt, dtype=np.float32),
            "qke": layers["qke"], "tsq": layers["tsq"],
            "qsq": layers["qsq"], "cov": layers["cov"],
        },
        closure=closure,
        bl_mynn_edmf_tke=bl_mynn_edmf_tke,
        tke_budget=tke_budget,
    )

    # ---- module_bl_mynn.F:1223-1233 dissipative heating, dheat_opt=1 -----
    diss_heat = np.zeros((ncol, nz), dtype=np.float32)
    for column in range(ncol):
        el = turbulence["el"][column]
        qke = predicted["qke"][column]
        for k in range(nz - 1):
            blend = max(F(F(0.5) * F(F(el[k]) + F(el[k + 1]))), F(1.0))
            value = F(F(F(F(1.0) * _powf(F(qke[k]), F(1.5)))
                        / F(B1 * blend)) / CP)
            value = min(max(value, F(0.0)), F(0.002))
            diss_heat[column, k] = F(
                value * _expf(
                    F(-F(F(10000.0) / max(F(layers["p"][column, k]), F(1.0))))
                )
            )

    # ---- module_bl_mynn.F:1237-1275 tendencies ---------------------------
    tendencies = mynn_tendencies_default(
        {
            "dz": layers["dz"], "rho": layers["rho"], "u": layers["u"],
            "v": layers["v"], "th": layers["th"], "tk": layers["tk"],
            "qv": qv1, "p": layers["p"], "exner": layers["exner"],
            "thl": thl, "sqv": layers["sqv"], "sqc": layers["sqc"],
            "sqi": layers["sqi"], "sqs": kzero, "ozone": zero_column,
            "tcd": turbulence["tcd"], "qcd": turbulence["qcd"],
            "dfm": turbulence["dfm"], "dfh": turbulence["dfh"],
            "diss_heat": diss_heat,
            "sub_thl": plumes["sub_thl"], "sub_sqv": plumes["sub_sqv"],
            "sub_u": plumes["sub_u"], "sub_v": plumes["sub_v"],
            "det_thl": plumes["det_thl"], "det_sqv": plumes["det_sqv"],
            "det_sqc": plumes["det_sqc"], "det_u": plumes["det_u"],
            "det_v": plumes["det_v"],
            "s_aw": plumes["s_aw"], "s_awthl": plumes["s_awthl"],
            "s_awqv": plumes["s_awqv"], "s_awqc": plumes["s_awqc"],
            "s_awu": plumes["s_awu"], "s_awv": plumes["s_awv"],
            "sd_aw": zero_interface, "sd_awthl": zero_interface,
            "sd_awqv": zero_interface, "sd_awqc": zero_interface,
            "sd_awu": zero_interface, "sd_awv": zero_interface,
            "delt": np.full(ncol, delt, dtype=np.float32),
            "psfc": scalars["ps"], "ust": scalars["ust"],
            "wspd": scalars["wspd"], "uoce": scalars["uoce"],
            "voce": scalars["voce"], "flt": flt, "flqv": flqv,
            "flqc": flqc,
            # Stage B: the five qn columns + the s_awqn* interfaces DMP_mf
            # just accumulated, exactly the pairing WRF binds at the
            # mynn_tendencies call.  Empty under mixscalars=0.
            **qn_layers,
            **({f"s_aw{name}": plumes[f"s_aw{name}"]
                for name in MYNN_TENDENCIES_QN_LAYER_INPUTS}
               if bl_mynn_mixscalars == 1 else {}),
        },
        bl_mynn_cloudmix=bl_mynn_cloudmix,
        bl_mynn_mixqt=bl_mynn_mixqt,
        bl_mynn_edmf=bl_mynn_edmf,
        bl_mynn_edmf_mom=bl_mynn_edmf_mom,
        bl_mynn_mixscalars=bl_mynn_mixscalars,
        flag_qs=flag_qs,
        flag_qnc=flag_qnc, flag_qni=flag_qni, flag_qnwfa=flag_qnwfa,
        flag_qnifa=flag_qnifa, flag_qnbca=flag_qnbca,
    )
    exchange = mynn_retrieve_exchange_coeffs({
        "dz": layers["dz"], "dfm": turbulence["dfm"],
        "dfh": turbulence["dfh"],
    })

    # ---- module_bl_mynn.F:1311-1355 write-back ---------------------------
    return {
        "rublten": tendencies["du"], "rvblten": tendencies["dv"],
        "rthblten": tendencies["dth"], "rqvblten": tendencies["dqv"],
        "rqcblten": tendencies["dqc"], "rqiblten": tendencies["dqi"],
        "rqsblten": tendencies["dqs"], "dozone": tendencies["dozone"],
        # Stage B: the five qn tendencies under WRF's RQN*BLTEN names.
        # Under mixscalars=0 these are the solver's structural zeros, and
        # no admitted consumer reads them -- additive keys, not new math.
        "rqncblten": tendencies["dqnc"],
        "rqniblten": tendencies["dqni"],
        "rqnwfablten": tendencies["dqnwfa"],
        "rqnifablten": tendencies["dqnifa"],
        "rqnbcablten": tendencies["dqnbca"],
        "exch_h": exchange["k_h"], "exch_m": exchange["k_m"],
        "qke": predicted["qke"], "tsq": predicted["tsq"],
        **({"qke_adv": predicted["qke"].copy()} if bl_mynn_tkeadvect else {}),
        "qsq": predicted["qsq"], "cov": predicted["cov"],
        "el": turbulence["el"], "sh": turbulence["sh"],
        "sm": turbulence["sm"], "qc_bl": qc_bl, "qi_bl": qi_bl,
        "cldfra_bl": cldfra_bl, "pblh": scalars["pblh"], "kpbl": kpbl,
        "rmol": rmol, "maxwidth": plumes["maxwidth"],
        "maxmf": plumes["maxmf"], "ztop_plume": plumes["ztop"],
        "ktop_plume": plumes["ktop"],
    }
