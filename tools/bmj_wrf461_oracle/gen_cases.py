"""Write the BMJ column-oracle input set (cases.bin) for run_bmj.F90.

The columns are synthetic soundings, built in binary64 and rounded to
float32 once, chosen to reach every branch of WRF v4.6.1 module_cu_bmj.F:
deep convection over land and over sea, shallow-only columns capped by an
inversion, stable columns that never convect, columns sitting on a moist
adiabat (the CAPE = 0 edge), high-topped deep columns, very low LCLs,
elevated terrain, and cold-season columns whose freezing level is inside
the cloud (the deep branch's ``T(L+1) < TFRZ`` exit to label 430).

Nothing here is BMJ arithmetic: it only has to produce plausible,
hydrostatically consistent WRF-shaped columns.  The Fortran harness reads
these bytes, and the CUDA side is later fed the very same bytes back from
the oracle's per-step input dump, so this generator never has to be
reproduced bit for bit anywhere.

Layout of cases.bin (little-endian, stream):
    int32 ncol, int32 nz
    float32 pint[ncol][nz+1]   interface pressure, bottom-up, Pa
    float32 pmid[ncol][nz]     mass-level pressure, Pa
    float32 t[ncol][nz]        temperature, K
    float32 qv[ncol][nz]       water-vapour mixing ratio, kg/kg
    float32 rho[ncol][nz]      density, kg/m3
    float32 dz8w[ncol][nz]     layer depth, m
    float32 xland[ncol]        1 land, 2 water
    int32   kpbl[ncol]         PBL top level (WRF index, read but unused)
    int32   family[ncol]       case-family id (see FAMILIES)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

RD = 287.0
CP = 7.0 * 287.0 / 2.0
G = 9.81
RV = 461.6
EPS = RD / RV

FAMILIES = (
    "deep-land", "deep-sea", "shallow-only", "stable", "moist-adiabat-edge",
    "high-top", "low-lcl", "elevated-terrain", "cold-season",
    "marginal-trigger", "deep-sea-dry-aloft", "shallow-moist-aloft",
)

NZ = 45
PTOP = 5000.0


def esat(t):
    # Bolton (1980) over water, Pa.
    tc = t - 273.15
    return 611.2 * math.exp(17.67 * tc / (tc + 243.5))


def qsat(t, p):
    e = esat(t)
    return EPS * e / max(p - e, 1.0)


def eta_levels(nz):
    # Stretched eta: fine near the ground, coarse aloft (WRF-like).
    x = np.linspace(0.0, 1.0, nz + 1)
    eta = 1.0 - (0.35 * x + 0.65 * x ** 1.6)
    eta[0], eta[-1] = 1.0, 0.0
    return eta


def moist_adiabat(t0, p_levels):
    """Pseudo-adiabat from a saturated parcel at p_levels[0], binary64."""
    out = [t0]
    t = t0
    for k in range(1, len(p_levels)):
        p0, p1 = p_levels[k - 1], p_levels[k]
        n = 20
        for j in range(n):
            pa = p0 + (p1 - p0) * j / n
            dp = (p1 - p0) / n
            lv = 2.5e6
            qs = qsat(t, pa)
            num = (RD * t / pa) * (1 + lv * qs / (RD * t))
            den = CP * (1 + (lv * lv * qs * EPS) / (CP * RD * t * t))
            t = t + num / den * dp
        out.append(t)
    return out


def column(rng, family, idx):
    eta = eta_levels(NZ)
    psfc = 100500.0 + rng.uniform(-1500, 1200)
    t0 = 300.0
    lapse = 6.8e-3
    z_trop = 12500.0
    rh_bl = 0.8
    rh_mid = 0.6
    rh_top = 0.2
    bl_top = 1200.0
    inv_z = None
    inv_dt = 0.0
    xland = 1.0
    adiabat = False
    if family == "deep-land":
        t0 = rng.uniform(301, 307); lapse = rng.uniform(6.6e-3, 7.6e-3)
        rh_bl = rng.uniform(0.75, 0.9); rh_mid = rng.uniform(0.5, 0.8)
        z_trop = rng.uniform(11500, 13500)
    elif family == "deep-sea":
        xland = 2.0
        t0 = rng.uniform(299, 302.5); lapse = rng.uniform(6.2e-3, 7.0e-3)
        rh_bl = rng.uniform(0.8, 0.92); rh_mid = rng.uniform(0.6, 0.85)
        z_trop = rng.uniform(13000, 15000)
    elif family == "shallow-only":
        t0 = rng.uniform(292, 299); lapse = rng.uniform(6.5e-3, 8.0e-3)
        rh_bl = rng.uniform(0.8, 0.95); rh_mid = rng.uniform(0.1, 0.3)
        inv_z = rng.uniform(1500, 2800); inv_dt = rng.uniform(2.0, 6.0)
        xland = float(rng.choice([1.0, 2.0]))
    elif family == "stable":
        t0 = rng.uniform(270, 288); lapse = rng.uniform(2.5e-3, 5.0e-3)
        rh_bl = rng.uniform(0.2, 0.6); rh_mid = rng.uniform(0.1, 0.4)
        xland = float(rng.choice([1.0, 2.0]))
    elif family == "moist-adiabat-edge":
        adiabat = True
        t0 = rng.uniform(290, 300)
        xland = float(rng.choice([1.0, 2.0]))
    elif family == "high-top":
        t0 = rng.uniform(303, 308); lapse = rng.uniform(7.0e-3, 8.0e-3)
        rh_bl = rng.uniform(0.8, 0.92); rh_mid = rng.uniform(0.6, 0.85)
        z_trop = rng.uniform(15500, 17500)
    elif family == "low-lcl":
        t0 = rng.uniform(295, 301); lapse = rng.uniform(6.0e-3, 7.0e-3)
        rh_bl = rng.uniform(0.95, 0.995); rh_mid = rng.uniform(0.7, 0.95)
        bl_top = rng.uniform(300, 900)
        xland = float(rng.choice([1.0, 2.0]))
    elif family == "elevated-terrain":
        psfc = rng.uniform(78000, 86000)
        t0 = rng.uniform(295, 303); lapse = rng.uniform(7.5e-3, 9.0e-3)
        rh_bl = rng.uniform(0.5, 0.75); rh_mid = rng.uniform(0.4, 0.7)
    elif family == "cold-season":
        t0 = rng.uniform(274, 281); lapse = rng.uniform(6.8e-3, 8.0e-3)
        rh_bl = rng.uniform(0.85, 0.97); rh_mid = rng.uniform(0.6, 0.9)
        z_trop = rng.uniform(9000, 11000)
        xland = float(rng.choice([1.0, 2.0]))
    elif family == "marginal-trigger":
        t0 = rng.uniform(296, 300); lapse = rng.uniform(5.8e-3, 6.4e-3)
        rh_bl = rng.uniform(0.7, 0.85); rh_mid = rng.uniform(0.4, 0.6)
        xland = float(rng.choice([1.0, 2.0]))
    elif family == "deep-sea-dry-aloft":
        xland = 2.0
        t0 = rng.uniform(300, 303); lapse = rng.uniform(6.8e-3, 7.4e-3)
        rh_bl = rng.uniform(0.85, 0.95); rh_mid = rng.uniform(0.15, 0.35)
    elif family == "shallow-moist-aloft":
        t0 = rng.uniform(294, 298); lapse = rng.uniform(7.0e-3, 8.5e-3)
        rh_bl = rng.uniform(0.85, 0.95); rh_mid = rng.uniform(0.7, 0.9)
        inv_z = rng.uniform(2500, 4000); inv_dt = rng.uniform(1.0, 3.0)
    pint = PTOP + eta * (psfc - PTOP)
    pmid = 0.5 * (pint[:-1] + pint[1:])
    t = np.empty(NZ)
    qv = np.empty(NZ)
    if adiabat:
        tad = moist_adiabat(t0, list(pmid))
        for k in range(NZ):
            tk = max(tad[k], 200.0)
            t[k] = tk + rng.uniform(-0.25, 0.25)
            rh = 1.0 if pmid[k] > 30000.0 else 0.3
            qv[k] = rh * qsat(t[k], pmid[k])
    else:
        # integrate heights as we go so the lapse rate is in z
        z = 0.0
        tv_prev = None
        zint = 0.0
        for k in range(NZ):
            if k == 0:
                z = 0.5 * (RD * t0 / G) * math.log(pint[0] / pint[1])
            else:
                z = z + (RD * tv_prev / G) * math.log(pmid[k - 1] / pmid[k])
            if z < z_trop:
                tk = t0 - lapse * z
            else:
                tk = t0 - lapse * z_trop + 1.5e-3 * (z - z_trop)
            if inv_z is not None and z > inv_z:
                tk += inv_dt
            tk = max(tk, 195.0)
            if z < bl_top:
                rh = rh_bl
            elif z < 6000.0:
                f = (z - bl_top) / (6000.0 - bl_top)
                rh = rh_bl + (rh_mid - rh_bl) * f
            elif z < z_trop:
                f = (z - 6000.0) / max(z_trop - 6000.0, 1.0)
                rh = rh_mid + (rh_top - rh_mid) * f
            else:
                rh = 0.05
            if inv_z is not None and z > inv_z and family == "shallow-only":
                rh = min(rh, rh_mid)
            t[k] = tk
            qv[k] = max(rh * qsat(tk, pmid[k]), 1.0e-7)
            tv_prev = tk * (1 + 0.608 * qv[k] / (1 + qv[k]))
    tv = t * (1 + qv / EPS) / (1 + qv)
    rho = pmid / (RD * tv)
    dz = (RD * tv / G) * np.log(pint[:-1] / pint[1:])
    kpbl = int(rng.integers(3, 12))
    return pint, pmid, t, qv, rho, dz, xland, kpbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cols = []
    fam_ids = []
    for fid, fam in enumerate(FAMILIES):
        for i in range(args.per_family):
            cols.append(column(rng, fam, i))
            fam_ids.append(fid)
    ncol = len(cols)
    f32 = np.float32
    pint = np.array([c[0] for c in cols], dtype=f32)
    pmid = np.array([c[1] for c in cols], dtype=f32)
    t = np.array([c[2] for c in cols], dtype=f32)
    qv = np.array([c[3] for c in cols], dtype=f32)
    rho = np.array([c[4] for c in cols], dtype=f32)
    dz = np.array([c[5] for c in cols], dtype=f32)
    xland = np.array([c[6] for c in cols], dtype=f32)
    kpbl = np.array([c[7] for c in cols], dtype=np.int32)
    fam = np.array(fam_ids, dtype=np.int32)
    with open(args.out, "wb") as fh:
        np.array([ncol, NZ], dtype="<i4").tofile(fh)
        for a in (pint, pmid, t, qv, rho, dz, xland):
            a.astype("<f4").tofile(fh)
        kpbl.astype("<i4").tofile(fh)
        fam.astype("<i4").tofile(fh)
    print(f"wrote {ncol} columns x {NZ} levels to {args.out}")


if __name__ == "__main__":
    main()
