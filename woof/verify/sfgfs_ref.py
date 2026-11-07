"""Scalar CPU reference for the GFS surface layer (``sf_sfclay_physics=3``).

A statement-for-statement binary64 transcription of WRF v4.6.1
``phys/module_sf_gfs.F`` (SF_GFS and the executed part of PROGTM),
``module_gfs_funcphys.F`` (GPVS/FPVS/FPVSX) and the FRACTIONAL_SEAICE
wrapper of ``phys/module_surface_driver.F``, written beside
``woof/core/kernels/sf_gfs.cu`` so the two can be read against each other
and against the Fortran.

It is verification tooling, not a forecast path.  Python floats are IEEE
binary64 with one rounding per operation, which is what the gfortran build
and the kernel's pinned intrinsics compute; float32 statements of SF_GFS go
through :class:`numpy.float32`.  ``exp``, ``log`` and ``pow`` come from
:mod:`math`, i.e. the HOST C library: on a glibc host those are the words
the oracle's gfortran build called, elsewhere they need not be, so
:func:`host_libm_is_oracle_libm` says which.  The kernel does not depend on
the host at all (its routines are WOOF's own, glibc_flt64.cuh).
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass

import numpy as np

f32 = np.float32


def F(x) -> float:
    """A default-kind (float32) Fortran literal, widened to binary64."""
    return float(f32(x))


def host_libm_is_oracle_libm() -> bool:
    """True where :mod:`math` is glibc's exp/log/pow, the oracle's libm."""
    return platform.system() == "Linux" and platform.libc_ver()[0] == "glibc"


# ---- module_gfs_physcons.F ----------------------------------------------------
CON_G = F(9.80665e+0)
CON_RD = F(2.8705e+2)
CON_RV = F(4.6150e+2)
CON_CVAP = F(1.8460e+3)
CON_CLIQ = F(4.1855e+3)
CON_CSOL = F(2.1060e+3)
CON_HVAP = F(2.5000e+6)
CON_HFUS = F(3.3358e+5)
CON_PSAT = F(6.1078e+2)
CON_TTP = F(2.7316e+2)
CON_FVIRT = CON_RV / CON_RD - 1.0
CON_EPS = CON_RD / CON_RV
CON_EPSM1 = CON_RD / CON_RV - 1.0

NXPVS = 7501


def fpvsx(t: float) -> float:
    """module_gfs_funcphys.F FPVSX: saturation vapour pressure (Pa)."""
    tliq = CON_TTP
    tice = CON_TTP - 20.0
    dldtl = CON_CVAP - CON_CLIQ
    heatl = CON_HVAP
    xponal = -dldtl / CON_RV
    xponbl = -dldtl / CON_RV + heatl / (CON_RV * CON_TTP)
    dldti = CON_CVAP - CON_CSOL
    heati = CON_HVAP + CON_HFUS
    xponai = -dldti / CON_RV
    xponbi = -dldti / CON_RV + heati / (CON_RV * CON_TTP)
    tr = CON_TTP / t
    if t >= tliq:
        return CON_PSAT * math.pow(tr, xponal) * math.exp(xponbl * (1.0 - tr))
    if t < tice:
        return CON_PSAT * math.pow(tr, xponai) * math.exp(xponbi * (1.0 - tr))
    w = (t - tice) / (tliq - tice)
    pvl = CON_PSAT * math.pow(tr, xponal) * math.exp(xponbl * (1.0 - tr))
    pvi = CON_PSAT * math.pow(tr, xponai) * math.exp(xponbi * (1.0 - tr))
    return w * pvl + (1.0 - w) * pvi


def gpvs() -> np.ndarray:
    """GPVS: the 7501 table words, then c1xpvs and c2xpvs (7503 binary64)."""
    xmin, xmax = 180.0, 330.0
    xinc = (xmax - xmin) / float(NXPVS - 1)
    c2 = 1.0 / xinc
    c1 = 1.0 - xmin * c2
    words = [fpvsx(xmin + float(j) * xinc) for j in range(NXPVS)]
    return np.asarray(words + [c1, c2], dtype=np.float64)


def fpvs(table: np.ndarray, t: float) -> float:
    """FPVS: linear interpolation in :func:`gpvs`'s table."""
    c1, c2 = float(table[NXPVS]), float(table[NXPVS + 1])
    xj = min(max(c1 + c2 * t, 1.0), float(NXPVS))
    jx = int(min(xj, float(NXPVS - 1)))
    lo, hi = float(table[jx - 1]), float(table[jx])
    return lo + (xj - float(jx)) * (hi - lo)


@dataclass(frozen=True)
class WrfConstants:
    """The REAL arguments the surface driver passes (module_model_constants)."""

    cp: np.float32
    rovcp: np.float32
    r: np.float32
    xlv: np.float32
    ep1: np.float32
    ep2: np.float32
    karman: np.float32


def wrf_constants() -> WrfConstants:
    """module_model_constants.F's values, folded in float32 as gfortran does."""
    r_d = f32(287.0)
    r_v = f32(461.6)
    cp = f32(f32(f32(7.0) * r_d) / f32(2.0))
    return WrfConstants(
        cp=cp, rovcp=f32(r_d / cp), r=r_d, xlv=f32(2.5e6),
        ep1=f32(f32(r_v / r_d) - f32(1.0)), ep2=f32(r_d / r_v),
        karman=f32(0.4))


OUTPUTS = ("br", "chs", "chs2", "cpm", "cqs2", "flhc", "flqc", "gz1oz0",
           "hfx", "lh", "psim", "psih", "qfx", "qgh", "qsfc", "u10", "v10",
           "wspd")


def sf_gfs_column(col: dict, znt: np.float32, ust: np.float32,
                  k: WrfConstants, table: np.ndarray, *, isfflx: int = 1,
                  hfx_in: np.float32 = f32(0.0)):
    """One column of SF_GFS.  Returns (outputs dict, znt, ust), float32."""
    u1, v1 = float(col["u1"]), float(col["v1"])
    t1 = float(col["t1"])
    qv1 = f32(col["qv1"])
    xland = f32(col["xland"])
    prsl1 = float(f32(f32(col["p1"]) * f32(.001)))
    ps = float(f32(f32(col["psfc"]) * f32(.001)))
    q1 = float(qv1)
    slimsk = float(abs(f32(xland - f32(2.0))))
    tskin = float(col["tsk"])
    ustar = float(ust)
    z0rl = float(f32(f32(znt) * f32(100.0)))
    rovcp = float(k.rovcp)
    prslki = math.pow(ps / prsl1, rovcp)
    thgb = tskin * math.pow(F(100.0) / ps, rovcp)
    thx = t1 * math.pow(F(100.0) / prsl1, rovcp)
    rho1 = prsl1 * F(1000.0) / (float(k.r) * t1 * (1.0 + float(k.ep1) * q1))
    q1 = q1 / (1.0 + q1)

    # PROGTM, executed statements only.
    xrcl = math.sqrt(1.0)
    psurf = F(1000.0) * ps
    ps1 = F(1000.0) * prsl1
    wind = xrcl * math.sqrt(u1 * u1 + v1 * v1) + max(0.0, min(0.0, 30.0))
    wind = max(wind, 1.0)
    q0 = max(q1, 1.0e-8)
    theta1 = t1 * prslki
    tv1 = t1 * (1.0 + CON_FVIRT * q0)
    thv1 = theta1 * (1.0 + CON_FVIRT * q0)
    tvs = tskin * (1.0 + CON_FVIRT * q0)
    qss = fpvs(table, tskin)
    qss = CON_EPS * qss / (psurf + CON_EPSM1 * qss)
    z0 = F(.01) * z0rl
    g = CON_G
    z1 = -CON_RD * tv1 * math.log(ps1 / psurf) / g
    charnock, ca, vis = F(.014), F(.4), F(1.4e-5)
    if slimsk == 0.0:
        ustar = math.sqrt(g * z0 / charnock)
    z0max = min(z0, F(0.1) * z1)
    ztmax = z0max
    if slimsk == 0.0:
        restar = ustar * z0max / vis
        restar = max(restar, .000001)
        rat = F(2.67) * math.pow(restar, F(.25)) - F(2.57)
        rat = min(rat, 7.0)
        ztmax = z0max * math.exp(-rat)
    dtv = thv1 - tvs
    adtv = max(abs(dtv), .001)
    dtv = math.copysign(1.0, dtv) * adtv
    rb = g * dtv * z1 / (F(.5) * (thv1 + tvs) * wind * wind)
    rb = max(rb, -5000.0)
    fm = math.log(z1 / z0max)
    fh = math.log(z1 / ztmax)
    hlinf = rb * fm * fm / fh
    fm10 = math.log((z0max + F(10.0)) / z0max)
    fh2 = math.log((ztmax + F(2.0)) / ztmax)
    alpha4 = F(4.0) * F(5.0)

    def stable_psi(a0, a):
        return a0 - a + math.log((a + 1.0) / (a0 + 1.0))

    if dtv >= 0.0:
        hl1 = hlinf
        if hlinf > F(.25):
            hl0inf = z0max * hlinf / z1
            hltinf = ztmax * hlinf / z1
            aa = math.sqrt(1.0 + alpha4 * hlinf)
            aa0 = math.sqrt(1.0 + alpha4 * hl0inf)
            bb = aa
            bb0 = math.sqrt(1.0 + alpha4 * hltinf)
            pm = stable_psi(aa0, aa)
            ph = stable_psi(bb0, bb)
            fms = fm - pm
            fhs = fh - ph
            hl1 = fms * fms * rb / fhs
        hl0 = z0max * hl1 / z1
        hlt = ztmax * hl1 / z1
        aa = math.sqrt(1.0 + alpha4 * hl1)
        aa0 = math.sqrt(1.0 + alpha4 * hl0)
        bb = aa
        bb0 = math.sqrt(1.0 + alpha4 * hlt)
        pm = stable_psi(aa0, aa)
        ph = stable_psi(bb0, bb)
        hl110 = hl1 * F(10.0) / z1
        aa = math.sqrt(1.0 + alpha4 * hl110)
        pm10 = stable_psi(aa0, aa)
        hl12 = hl1 * F(2.0) / z1
        bb = math.sqrt(1.0 + alpha4 * hl12)
        ph2 = stable_psi(bb0, bb)
    else:
        olinf = z1 / hlinf
        if abs(olinf) <= F(50.0) * z0max:
            hlinf = -z1 / (F(50.0) * z0max)
        a0, a1, b1, b2 = F(-3.975), F(12.32), F(-7.755), F(6.041)
        a0p, a1p, b1p, b2p = F(-7.941), F(24.75), F(-8.705), F(7.899)
        pm = ph = pm10 = ph2 = 0.0
        if hlinf >= F(-.5):
            hl1 = hlinf
            pm = (a0 + a1 * hl1) * hl1 / (1.0 + b1 * hl1 + b2 * hl1 * hl1)
            ph = (a0p + a1p * hl1) * hl1 / (1.0 + b1p * hl1 + b2p * hl1 * hl1)
            hl110 = hl1 * F(10.0) / z1
            pm10 = ((a0 + a1 * hl110) * hl110
                    / (1.0 + b1 * hl110 + b2 * hl110 * hl110))
            hl12 = hl1 * F(2.0) / z1
            ph2 = ((a0p + a1p * hl12) * hl12
                   / (1.0 + b1p * hl12 + b2p * hl12 * hl12))
        if hlinf < F(-.5):
            hl1 = -hlinf
            pm = math.log(hl1) + F(2.0) * math.pow(hl1, F(-.25)) - F(.8776)
            ph = math.log(hl1) + F(.5) * math.pow(hl1, F(-.5)) + F(1.386)
            hl110 = hl1 * F(10.0) / z1
            pm10 = (math.log(hl110) + F(2.0) * math.pow(hl110, F(-.25))
                    - F(.8776))
            hl12 = hl1 * F(2.0) / z1
            ph2 = (math.log(hl12) + F(.5) * math.pow(hl12, F(-.5))
                   + F(1.386))
    fm = fm - pm
    fh = fh - ph
    fm10 = fm10 - pm10
    fh2 = fh2 - ph2
    cm = ca * ca / (fm * fm)
    ch = ca * ca / (fm * fh)
    stress = cm * wind * wind
    ustar = math.sqrt(stress)
    if slimsk == 0.0:
        z0 = (charnock / g) * (ustar * ustar)
        z0 = min(z0, .1)
        z0 = max(z0, 1.0e-7)
        z0rl = F(100.0) * z0
    f10m = min(fm10 / fm, 1.0)
    u10m = f10m * xrcl * u1
    v10m = f10m * xrcl * v1

    out = {}
    out["u10"] = f32(u10m)
    out["v10"] = f32(v10m)
    out["br"] = f32(rb)
    out["chs"] = f32(ch * wind)
    out["chs2"] = f32(ustar * float(k.karman) / fh2)
    out["cpm"] = f32(k.cp * f32(f32(1.0) + f32(f32(.8) * qv1)))
    esat = f32(fpvs(table, t1))
    out["qgh"] = f32(float(f32(k.ep2 * esat)) / (F(1000.0) * ps - float(esat)))
    out["qsfc"] = f32(qss)
    out["psih"] = f32(ph)
    out["psim"] = f32(pm)
    ust_out = f32(ustar)
    out["wspd"] = f32(wind)
    znt_out = f32(z0rl * F(.01))
    out["flhc"] = f32(float(out["cpm"]) * rho1 * float(out["chs"]))
    out["flqc"] = f32(rho1 * float(out["chs"]))
    out["gz1oz0"] = f32(math.log(z1 / (z0rl * F(.01))))
    out["cqs2"] = out["chs2"]
    if isfflx == 0:
        out["hfx"] = out["lh"] = out["qfx"] = f32(0.0)
    else:
        dx = f32(xland - f32(1.5))
        hfx = hfx_in
        if dx > 0:
            hfx = f32(float(out["flhc"]) * (thgb - thx))
        elif dx < 0:
            hfx = f32(float(out["flhc"]) * (thgb - thx))
            hfx = hfx if hfx > f32(-250.0) else f32(-250.0)
        out["hfx"] = hfx
        qfx = f32(float(out["flqc"]) * (float(out["qsfc"]) - q1))
        out["qfx"] = qfx if qfx > f32(0.0) else f32(0.0)
        out["lh"] = f32(k.xlv * out["qfx"])
    return out, znt_out, ust_out


SEA_CARRIERS = ("chs_sea", "chs2_sea", "cpm_sea", "cqs2_sea", "flhc_sea",
                "flqc_sea", "hfx_sea", "lh_sea", "qfx_sea", "qgh_sea",
                "qsfc_sea", "ust_sea", "znt_sea")


def sf_gfs_seaice_column(col: dict, state: dict, k: WrfConstants,
                         table: np.ndarray, *, itimestep: int,
                         xice_threshold: np.float32 = f32(0.02),
                         xland_sea_fix: bool = True, isfflx: int = 1):
    """The FRACTIONAL_SEAICE wrapper for one column.

    ``state`` holds the carried znt, ust, sst and every *_sea carrier and is
    updated in place; returns the grid outputs and the open-water TSK_SEA.
    ``xland_sea_fix=False`` reproduces WRF's shipped open-water call (XLAND
    instead of XLAND_SEA); WOOF runs ``True``.
    """
    xi = f32(col["xice"])
    tsk = f32(col["tsk"])
    ice = bool(xi >= xice_threshold and xi <= f32(1.0))
    s = f32(state["sst"])
    if ice:
        if s < f32(271.4):
            s = f32(271.4)
        if s > f32(273.0) and itimestep <= 3:
            if xi >= f32(0.6):
                s = f32(271.4)
            elif xi >= f32(0.4):
                s = f32(273.0)
            elif xi >= f32(0.2) and s > f32(275.0):
                s = f32(275.0)
            elif s > f32(278.0):
                s = f32(278.0)
        tsk_ice = f32(f32(tsk - f32(f32(f32(1.0) - xi) * s)) / xi)
        if tsk_ice < f32(221.4):
            tsk_ice = f32(221.4)
        if xi < f32(0.2) and tsk < f32(253.15):
            tsk_ice = f32(253.15)
        if xi < f32(0.1) and tsk < f32(263.15):
            tsk_ice = f32(263.15)
    else:
        tsk_ice = tsk
    znt_hold, ust_hold = f32(state["znt"]), f32(state["ust"])
    first = dict(col, tsk=tsk_ice)
    out, znt1, ust1 = sf_gfs_column(first, znt_hold, ust_hold, k, table,
                                    isfflx=isfflx, hfx_in=f32(state["hfx"]))
    if ice:
        xland_sea = f32(2.0)
        zs = f32(0.0001)
        if s < f32(271.4):
            s = f32(271.4)
        tsk_sea = s
        us = f32(state["ust_sea"])
    else:
        xland_sea = f32(col["xland"])
        zs = znt_hold
        us = ust_hold
        tsk_sea = tsk
    sea = dict(col, tsk=tsk_sea,
               xland=(xland_sea if xland_sea_fix else f32(col["xland"])))
    out_sea, zs, us = sf_gfs_column(sea, zs, us, k, table, isfflx=isfflx,
                                    hfx_in=f32(state["hfx_sea"]))
    if ice:
        omx = f32(f32(1.0) - xi)
        for name in ("br", "gz1oz0", "psim", "psih", "u10", "v10", "wspd"):
            out[name] = f32(f32(out[name] * xi) + f32(omx * out_sea[name]))
    cpm_first = out["cpm"]
    out["cpm"] = out_sea["cpm"]
    state["znt"], state["ust"] = znt1, ust1
    for name in ("chs", "chs2", "cqs2", "flhc", "flqc", "hfx", "lh", "qfx",
                 "qgh", "qsfc"):
        state[f"{name}_sea"] = out_sea[name]
    state["cpm_sea"] = cpm_first
    state["ust_sea"], state["znt_sea"] = us, zs
    state["hfx"] = out["hfx"]
    if ice:
        state["sst"] = s
    return out, tsk_sea


__all__ = ["CON_EPS", "CON_EPSM1", "CON_FVIRT", "NXPVS", "OUTPUTS",
           "SEA_CARRIERS", "WrfConstants", "fpvs", "fpvsx", "gpvs",
           "host_libm_is_oracle_libm", "sf_gfs_column",
           "sf_gfs_seaice_column", "wrf_constants"]
