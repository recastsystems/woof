"""NumPy reference for the GFS PBL (WRF v4.6.1 phys/module_bl_gfs.F).

A second, independent transcription of ``BL_GFS``/``MONINP``/``TRIDI2``/
``TRIDIN``/``TRIDIT``, vectorised over columns and written from the Fortran
statement by statement in its evaluation order: binary32 where every operand
is default REAL, binary64 (``kind_phys``) otherwise, one rounding per
operation (NumPy's scalar and array arithmetic is IEEE round-to-nearest).
The two real powers call the C library's scalar ``pow`` and ``powf``
element by element (not NumPy's ``power``, whose vector loops may dispatch
elsewhere) -- the same entry points the gfortran oracle calls -- so on the
oracle's host this reference is graded word for word
(tests/test_gfs_pbl_wrf461_parity.py).  The CUDA kernel is graded against
the oracle directly; this module is the CPU half of the check.

The binary64 constants are the generated words of
``woof/data/gfs_pbl/gfs_pbl_constants.cuh`` (parsed, not retyped).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

F = np.float32
D = np.float64
_HEADER = (Path(__file__).resolve().parents[1] / "data" / "gfs_pbl"
           / "gfs_pbl_constants.cuh")


@lru_cache(maxsize=1)
def constants() -> dict[str, np.float64]:
    words = {}
    for name, value in re.findall(r"#define (GFS_\w+) \(([^)]*)\)",
                                  _HEADER.read_text(encoding="utf-8")):
        words[name] = D(float.fromhex(value))
    return words


@lru_cache(maxsize=1)
def _libm():
    lib = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    lib.powf.restype = ctypes.c_float
    lib.powf.argtypes = (ctypes.c_float, ctypes.c_float)
    lib.pow.restype = ctypes.c_double
    lib.pow.argtypes = (ctypes.c_double, ctypes.c_double)
    return lib


def _powf(x: np.ndarray, y) -> np.ndarray:
    lib = _libm()
    return np.array([lib.powf(float(v), float(y)) for v in np.ravel(x)],
                    F).reshape(np.shape(x))


def _pow(x: np.ndarray, y) -> np.ndarray:
    lib = _libm()
    return np.array([lib.pow(float(v), float(y)) for v in np.ravel(x)],
                    D).reshape(np.shape(x))


def _max(a, b):
    """gfortran REAL(8) MAX(a, b): a where a > b, else b."""
    return np.where(a > b, a, b)


def _min(a, b):
    """gfortran REAL(8) MIN(a, b): b where b < a, else a."""
    return np.where(b < a, b, a)


def bl_gfs(inp: dict, *, dt: float, flqi: bool,
           wrf_constants: tuple) -> dict:
    """One BL_GFS call over ``ncol`` columns.

    ``inp`` holds u, v, t, qv, qc, qi, p, pi, dz8w, z as float32 ``(nz,
    ncol)`` and psfc, psim, psih, hfx, qfx, gz1oz0, wspd, br as float32
    ``(ncol,)``.  ``wrf_constants`` is (cp, g, rovcp, r, rovg, ep1, karman)
    as float32.  Returns the WRF outputs in the same layout.
    """
    c = constants()
    cp_, g_, rovcp, r_, rovg, ep1, karman = (F(x) for x in wrf_constants)
    u3, v3, t3 = (np.asarray(inp[n], F) for n in ("u", "v", "t"))
    qv3, qc3, qi3 = (np.asarray(inp[n], F) for n in ("qv", "qc", "qi"))
    p3, pi3 = np.asarray(inp["p"], F), np.asarray(inp["pi"], F)
    dz8w, z3 = np.asarray(inp["dz8w"], F), np.asarray(inp["z"], F)
    psfc, psim, psih = (np.asarray(inp[n], F) for n in ("psfc", "psim", "psih"))
    hfx, qfx = np.asarray(inp["hfx"], F), np.asarray(inp["qfx"], F)
    gz1oz0, br = np.asarray(inp["gz1oz0"], F), np.asarray(inp["br"], F)
    wspd = np.asarray(inp["wspd"], F)
    km, ncol = u3.shape
    with np.errstate(all="ignore"):
        # ---- BL_GFS :279-411 -------------------------------------------
        rrhox = D((r_ * t3[0] * (F(1.0) + ep1 * qv3[0])) / psfc)
        cpm = D(cp_ * (F(1.0) + F(0.8) * qv3[0]))
        fmtmp = D(gz1oz0 - psim)
        psk = D(_powf(psfc * F(0.00001), rovcp))
        fm = fmtmp
        fh = D(gz1oz0 - psih)
        heat = D(hfx) / cpm * rrhox
        evap = D(qfx) * rrhox
        stress = D(karman * karman * wspd * wspd) / (fmtmp * fmtmp)
        spd1 = D(wspd)
        rcl = D(1.0)
        rbsoil = D(br)
        u1, v1, t1 = D(u3), D(v3), D(t3)
        q1 = [D(qv3 / (F(1.0) + qv3)), D(qc3 / (F(1.0) + qc3))]
        if flqi:
            q1.append(D(qi3 / (F(1.0) + qi3)))
        prsl = D(p3 * F(0.001))
        prslk = _pow(prsl * D(F(0.01)), D(rovcp))
        prsi = np.empty((km + 1, ncol), D)
        phii = np.empty((km + 1, ncol), D)
        phil = np.empty((km, ncol), D)
        dl = np.empty((km, ncol), D)
        prsi[0] = D(psfc * F(0.001))
        phii[0] = 0.0
        for k in range(1, km):
            m = k - 1
            dl[m] = prsl[m] / D(rovg) * D(dz8w[m]) / D(t3[m])
            prsi[k] = prsi[m] - dl[m]
            phii[k] = D((z3[k] - z3[0]) * g_)
            phil[m] = D(F(0.5) * (z3[k] + z3[m] - F(2.0) * z3[0]) * g_)
        dl[km - 1] = dl[km - 2]
        prsi[km] = prsi[km - 1] - dl[km - 2]
        phii[km] = phii[km - 1] + D(dz8w[km - 1] * g_)
        phil[km - 1] = phii[km - 1] - phil[km - 2] + phii[km - 1]
        ntrac = 3 if flqi else 2
        deltim = D(F(dt))

        # ---- MONINP :729-1298 -------------------------------------------
        rbcr = D(0.25)
        gravi = c["GFS_GRAVI"]
        dt2 = D(2.0) * deltim
        rdt = D(1.0) / dt2
        kmpbl = km // 2
        zi = phii[:km] * gravi
        zl = phil * gravi
        theta = np.zeros((km, ncol), D)
        for k in range(kmpbl):
            theta[k] = t1[k] * psk / prslk[k]
        rdzt = np.empty((km - 1, ncol), D)
        for k in range(km - 1):
            rdzt[k] = c["GFS_GOR"] * prsi[k + 1] / (prsl[k] - prsl[k + 1])
        hgamt = np.zeros(ncol, D)
        hgamq = np.zeros(ncol, D)
        kpbl = np.ones(ncol, np.int64)
        hpbl = zi[1].copy()
        pblflg = np.ones(ncol, bool)
        sfcflg = ~(rbsoil > 0.0)
        rdzt1 = c["GFS_GOR"] * prsl[0] / dl[0]
        beta = dt2 * rdzt1 / t1[0]
        ustar = np.sqrt(stress)
        fv, qmin, gg = c["GFS_FV"], c["GFS_QMIN"], c["GFS_G"]
        the1 = theta[0]
        the1v = the1 * (D(1.0) + fv * _max(q1[0][0], qmin))
        thermal = the1v.copy()
        cols = np.arange(ncol)

        def pbl_search(active, excess):
            nonlocal kpbl, hpbl
            stable = np.zeros(ncol, bool)
            rbup = rbsoil.copy()
            rbdn = rbsoil.copy()
            for k in range(1, kmpbl):
                go = active & ~stable
                thekv = theta[k] * (D(1.0) + fv * _max(q1[0][k], qmin))
                spdk2 = _max(rcl * (u1[k] * u1[k] + v1[k] * v1[k]), D(1.0))
                up = (thekv - excess) * (gg * zl[k] / the1v) / spdk2
                rbdn = np.where(go, rbup, rbdn)
                rbup = np.where(go, up, rbup)
                kpbl = np.where(go, k + 1, kpbl)
                stable = np.where(go, rbup > rbcr, stable)
            k = kpbl - 1  # zero-based index of level KPBL
            rbint = np.where(rbdn >= rbcr, D(0.0), np.where(
                rbup <= rbcr, D(1.0), (rbcr - rbdn) / (rbup - rbdn)))
            zlk = zl[k, cols]
            zlm = zl[k - 1, cols]
            new = zlm + rbint * (zlk - zlm)
            hpbl = np.where(active, new, hpbl)
            drop = active & (hpbl < zi[kpbl - 1, cols])
            kpbl = np.where(drop, kpbl - 1, kpbl)

        pbl_search(np.ones(ncol, bool), the1v)
        hol = _max(rbsoil * fm * fm / fh, c["GFS_RIMIN"])
        hol = np.where(sfcflg, _min(hol, -c["GFS_ZFMIN"]),
                       _max(hol, c["GFS_ZFMIN"]))
        hol = hol * hpbl / zl[0] * c["GFS_SFCFRAC"]
        tem = D(1.0) / (D(1.0) - c["GFS_APHI16"] * hol)
        phih_u = np.sqrt(tem)
        phim_u = np.sqrt(phih_u)
        phim_s = D(1.0) + c["GFS_APHI5"] * hol
        phim = np.where(sfcflg, phim_u, phim_s)
        phih = np.where(sfcflg, phih_u, phim_s)
        wscale = ustar / phim
        wscale = _min(wscale, ustar * c["GFS_APHI16"])
        wscale = _max(wscale, ustar / c["GFS_APHI5"])
        sflux = heat + evap * fv * the1
        unst = sfcflg & (sflux > 0.0)
        ht = _min(c["GFS_CFAC"] * heat / wscale, c["GFS_GAMCRT"])
        hq = _min(c["GFS_CFAC"] * evap / wscale, c["GFS_GAMCRQ"])
        vpert = _min(ht + fv * the1 * hq, c["GFS_GAMCRT"])
        thermal = np.where(unst, thermal + _max(vpert, D(0.0)), thermal)
        hgamt = np.where(unst, _max(ht, D(0.0)), hgamt)
        hgamq = np.where(unst, _max(hq, D(0.0)), hgamq)
        pblflg = unst
        kpbl = np.where(pblflg, 1, kpbl)
        hpbl = np.where(pblflg, zi[1], hpbl)
        pbl_search(pblflg, thermal)
        pblflg = pblflg & ~(kpbl <= 1)

        dku = np.zeros((km - 1, ncol), D)
        dkt = np.zeros((km - 1, ncol), D)
        prinv = D(1.0) / (phih / phim + c["GFS_CFAC_VK_P1"])
        prinv = _max(_min(prinv, c["GFS_PRMAX"]), c["GFS_PRMIN"])
        for k in range(kmpbl):
            sel = kpbl > k + 1
            zfac = _max(D(1.0) - (zi[k + 1] - zl[0]) / (hpbl - zl[0]),
                        c["GFS_ZFMIN"])
            u_ = c["GFS_XKZO"] + wscale * c["GFS_VK"] * zi[k + 1] * D(1.0) \
                * (zfac * zfac)
            t_ = u_ * prinv
            u_ = _max(_min(u_, c["GFS_DKMAX"]), c["GFS_DKMIN"])
            t_ = _max(_min(t_, c["GFS_DKMAX"]), c["GFS_DKMIN"])
            dku[k] = np.where(sel, u_, dku[k])
            dkt[k] = np.where(sel, t_, dkt[k])
        for k in range(km - 1):
            sel = (k + 1) >= kpbl
            ti = D(2.0) / (t1[k] + t1[k + 1])
            rdz = rdzt[k] * ti
            du_ = u1[k] - u1[k + 1]
            dv_ = v1[k] - v1[k + 1]
            dw2 = rcl * (du_ * du_ + dv_ * dv_)
            shr2 = _max(dw2, c["GFS_DW2MIN"]) * (rdz * rdz)
            tvd = t1[k] * (D(1.0) + fv * _max(q1[0][k], qmin))
            tvu = t1[k + 1] * (D(1.0) + fv * _max(q1[0][k + 1], qmin))
            bvf2 = gg * (c["GFS_GOCP"] + rdz * (tvu - tvd)) * ti
            ri = _max(bvf2 / shr2, c["GFS_RIMIN"])
            zk = c["GFS_VK"] * zi[k + 1]
            rl2 = zk * c["GFS_RLAM"] / (c["GFS_RLAM"] + zk)
            dk = rl2 * rl2 * np.sqrt(shr2)
            sri = np.sqrt(-ri)
            u_un = c["GFS_XKZO"] + dk * (D(1.0) + c["GFS_F_8"] * (-ri)
                                         / (D(1.0) + c["GFS_F_1746"] * sri))
            tem_un = dk * (D(1.0) + c["GFS_F_8"] * (-ri)
                           / (D(1.0) + c["GFS_F_1286"] * sri))
            b = D(1.0) + c["GFS_F_5"] * ri
            tem_st = dk / (b * b)
            t_st = c["GFS_XKZO"] + tem_st
            prnum = _min(D(1.0) + c["GFS_F_21"] * ri, c["GFS_PRMAX"])
            u_st = (t_st - c["GFS_XKZO"]) * prnum + c["GFS_XKZO"]
            unstable = ri < 0.0
            u_ = np.where(unstable, u_un, u_st)
            t_ = np.where(unstable, c["GFS_XKZO"] + tem_un, t_st)
            u_ = _max(_min(u_, c["GFS_DKMAX"]), c["GFS_DKMIN"])
            t_ = _max(_min(t_, c["GFS_DKMAX"]), c["GFS_DKMIN"])
            dku[k] = np.where(sel, u_, dku[k])
            dkt[k] = np.where(sel, t_, dkt[k])

        def rows(k):
            dtodsd = dt2 / dl[k]
            dtodsu = dt2 / dl[k + 1]
            dsig = prsl[k] - prsl[k + 1]
            rdz = rdzt[k] * D(2.0) / (t1[k] + t1[k + 1])
            return dtodsd, dtodsu, dsig, rdz

        def solve(al, ad, au, rhs):
            """TRIDIN/TRIDI2/TRIDIT's elimination for every right-hand side."""
            au = au.copy()
            rhs = [r.copy() for r in rhs]
            fk = D(1.0) / ad[0]
            au[0] = fk * au[0]
            for r in rhs:
                r[0] = fk * r[0]
            for k in range(1, km - 1):
                fkk = D(1.0) / (ad[k] - al[k - 1] * au[k - 1])
                au[k] = fkk * au[k]
                for r in rhs:
                    r[k] = fkk * (r[k] - al[k - 1] * r[k - 1])
            fk = D(1.0) / (ad[km - 1] - al[km - 2] * au[km - 2])
            for r in rhs:
                r[km - 1] = fk * (r[km - 1] - al[km - 2] * r[km - 2])
            for k in range(km - 2, -1, -1):
                for r in rhs:
                    r[k] = r[k] - au[k] * r[k + 1]
            return rhs

        # Heat and moisture :1137-1193.
        ad = np.empty((km, ncol), D)
        al = np.empty((km - 1, ncol), D)
        au = np.empty((km - 1, ncol), D)
        a1 = np.empty((km, ncol), D)
        a2 = np.empty((km, ncol), D)
        ad[0] = 1.0
        a1[0] = t1[0] + beta * heat
        a2[0] = q1[0][0] + beta * evap
        for k in range(km - 1):
            dtodsd, dtodsu, dsig, rdz = rows(k)
            tem1 = dsig * dkt[k] * rdz
            pbl = pblflg & ((k + 1) < kpbl)
            tem = D(1.0) / hpbl
            dsdzt = np.where(pbl, tem1 * (c["GFS_GOCP"] - hgamt * tem),
                             tem1 * c["GFS_GOCP"])
            dsdzq = tem1 * (-(hgamq * tem))
            a2[k] = np.where(pbl, a2[k] + dtodsd * dsdzq, a2[k])
            a2[k + 1] = np.where(pbl, q1[0][k + 1] - dtodsu * dsdzq,
                                 q1[0][k + 1])
            dsdz2 = tem1 * rdz
            au[k] = -(dtodsd * dsdz2)
            al[k] = -(dtodsu * dsdz2)
            ad[k] = ad[k] - au[k]
            ad[k + 1] = D(1.0) - al[k]
            a1[k] = a1[k] + dtodsd * dsdzt
            a1[k + 1] = t1[k + 1] - dtodsu * dsdzt
        a1, a2 = solve(al, ad, au, [a1, a2])
        tau = D(0.0) + (a1 - t1) * rdt
        rtg = [D(0.0) + (a2 - q1[0]) * rdt]
        # Momentum :1197-1241.
        ad[0] = D(1.0) + beta * stress / spd1
        a1[0] = u1[0]
        a2[0] = v1[0]
        for k in range(km - 1):
            dtodsd, dtodsu, dsig, rdz = rows(k)
            dsdz2 = dsig * dku[k] * rdz * rdz
            au[k] = -(dtodsd * dsdz2)
            al[k] = -(dtodsu * dsdz2)
            ad[k] = ad[k] - au[k]
            ad[k + 1] = D(1.0) - al[k]
            a1[k + 1] = u1[k + 1]
            a2[k + 1] = v1[k + 1]
        a1, a2 = solve(al, ad, au, [a1, a2])
        du = D(0.0) + (a1 - u1) * rdt
        dv = D(0.0) + (a2 - v1) * rdt
        # Tracers :1246-1295.
        ad[0] = 1.0
        for k in range(km - 1):
            dtodsd, dtodsu, dsig, rdz = rows(k)
            tem1 = dsig * dkt[k] * rdz
            dsdz2 = tem1 * rdz
            au[k] = -(dtodsd * dsdz2)
            al[k] = -(dtodsu * dsdz2)
            ad[k] = ad[k] - au[k]
            ad[k + 1] = D(1.0) - al[k]
        ats = solve(al, ad, au, [q.copy() for q in q1[1:]])
        for q, at in zip(q1[1:], ats):
            rtg.append(D(0.0) + (at - q) * rdt)

        out = {
            "rublten": F(du), "rvblten": F(dv),
            "rthblten": F(tau / D(pi3)),
            "rqvblten": F(rtg[0] / ((D(1.0) - q1[0]) * (D(1.0) - q1[0]))),
            "rqcblten": F(rtg[1] / ((D(1.0) - q1[1]) * (D(1.0) - q1[1]))),
            "rqiblten": (F(rtg[2] / ((D(1.0) - q1[2]) * (D(1.0) - q1[2])))
                         if ntrac == 3 else np.zeros((km, ncol), F)),
            "ust": F(np.sqrt(stress)),
            "wspd": np.sqrt(u3[0] * u3[0] + v3[0] * v3[0]) + F(1.0e-9),
            "pbl": F(hpbl),
            "kpbl": kpbl.astype(np.int32),
        }
    return out
