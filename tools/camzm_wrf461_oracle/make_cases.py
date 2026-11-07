"""Write the column cases for the CAM Zhang-McFarlane (cu_physics = 7) oracle.

    python3 make_cases.py OUTDIR

Each case FILE is one WRF tile: one dx, dt, cudt, PBL arm and vertical grid,
because those are scheme-wide in WRF (zm_convi's tau is set from DX once,
get_tpert's arm from bl_pbl_physics).  Each COLUMN in a file is one sounding.
The soundings are synthetic and seeded, built hydrostatically on a WRF-like
dry-mass eta grid, and grouped into regime families:

  tropical    deep, moist, warm ocean column (xland = 2)
  severe      midlatitude continental: moist boundary layer under an
              elevated mixed layer, dry mid troposphere, strong shear
  elevated    cool stable surface layer under a warm moist layer
  marginal    moisture tuned near the scheme's CAPE threshold (capelmt = 70)
  dry         desert column, no moisture to trigger on
  terrain     high terrain (ht 1.8-3.2 km), monsoon moisture
  cold        cold-season column whose plume glaciates (snow production)
  stable      strong inversion everywhere: the no-trigger edge
  deeppbl     plume whose top lies inside a very deep PBL (no_deep_pbl)
  toptrop     a low model top (p_top 20 kPa) so the plume reaches the top
  moistsat    near-saturated column (evaporation limiter edge)
  pbltrap     a plume capped by an inversion inside a very deep PBL, so
              zm(jt) < pblh and no_deep_pbl zeroes the cloud-base flux

Nothing here is a case name in the engine's sense: the families describe
the column, not a place or a date.

File layout (little-endian stream, read by run_camzm.F90):
  int32  magic 'CZM7', version 1, ncol, nk, nsteps, bl_pbl_physics,
         sf_sfclay_physics, itimestep0, stepcu
  float32 dt, dx, cudt
  float32 mass arrays (nk, ncol) C order = (ncol, nk) Fortran order:
         th t p pi qv qc qi z dz8w u v cldfra
  float32 interface arrays (nk+1, ncol): p8w z_at_w tke_pbl
  float32 surface arrays (ncol): tsk ust mavail pblh xland ht psfc hfx qfx tpert
  int32  kpbl (ncol)
and a JSON index (cases-index.json) naming every column's family.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x435A4D37  # 'CZM7'
F32 = np.float32
RD, CP, G, RV = 287.0, 1004.0, 9.81, 461.6
RCP = RD / CP
EPS = RD / RV


def esat(tk):
    """Bolton saturation vapour pressure (Pa) over liquid."""
    tc = tk - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def eta_levels(nk, rng=None):
    """Monotone eta interfaces 1 -> 0 with a WRF-like low-level squeeze."""
    x = np.linspace(0.0, 1.0, nk + 1)
    eta = 1.0 - (0.62 * x + 0.38 * x ** 2.2)
    eta[0], eta[-1] = 1.0, 0.0
    return eta


def column(family, nk, ptop, rng):
    """Build one sounding.  Returns a dict of float64 profiles."""
    r = lambda lo, hi: rng.uniform(lo, hi)
    ht = 0.0
    xland = 1.0
    pblh = r(600, 1800)
    bump = None
    if family == "tropical":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(299, 302.5), r(0.80, 0.9), r(0.65, 0.85), 5.0e-3, 5.6e-3
        xland, ztrop, pblh = 2.0, r(15500, 16500), r(500, 900)
        bltop = r(800, 1200)
    elif family == "severe":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(301, 306), r(0.62, 0.78), r(0.2, 0.4), 6.0e-3, 8.3e-3
        ztrop, pblh, bltop = r(12500, 13500), r(1200, 2200), r(1200, 1800)
    elif family == "elevated":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(286, 291), r(0.7, 0.9), r(0.45, 0.65), -4.0e-3, 6.8e-3
        ztrop, pblh, bltop = r(11500, 12500), r(150, 400), r(900, 1400)
    elif family == "marginal":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(293, 297), r(0.55, 0.72), r(0.35, 0.6), 6.5e-3, 6.6e-3
        ztrop, pblh, bltop = r(11500, 12500), r(800, 1400), r(800, 1200)
    elif family == "dry":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(305, 312), r(0.05, 0.15), r(0.05, 0.15), 9.0e-3, 7.5e-3
        ztrop, pblh, bltop = r(13000, 14000), r(2500, 4000), r(2500, 3500)
    elif family == "terrain":
        ht = r(1800, 3200)
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(291, 297), r(0.65, 0.85), r(0.5, 0.75), 7.0e-3, 6.5e-3
        ztrop, pblh, bltop = r(15000, 16500), r(800, 1600), r(800, 1200)
    elif family == "cold":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(276, 281), r(0.85, 0.95), r(0.7, 0.9), 7.5e-3, 7.0e-3
        ztrop, pblh, bltop = r(9000, 10000), r(500, 900), r(600, 900)
    elif family == "stable":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(280, 288), r(0.6, 0.8), r(0.3, 0.5), -6.0e-3, 4.0e-3
        ztrop, pblh, bltop = r(10500, 11500), r(100, 300), r(500, 800)
    elif family == "deeppbl":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(302, 306), r(0.45, 0.6), r(0.05, 0.12), 9.5e-3, 9.0e-3
        ztrop, pblh, bltop = r(13000, 14000), r(5500, 7000), r(3000, 4000)
    elif family == "toptrop":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(300, 303), r(0.82, 0.9), r(0.7, 0.85), 5.0e-3, 5.6e-3
        xland, ztrop, pblh, bltop = 2.0, r(15500, 16500), r(500, 900), r(800, 1200)
    elif family == "pbltrap":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(301, 304), r(0.72, 0.82), r(0.4, 0.6), 7.0e-3, 6.8e-3
        ztrop, pblh, bltop = r(13000, 14000), r(7000, 8000), r(1200, 1600)
        bump = (r(9.0, 12.0), r(4200, 4800))
    elif family == "moistsat":
        tsfc, rh_bl, rh_mid, gam_low, gam_mid = r(298, 301), r(0.95, 0.99), r(0.95, 0.995), 5.2e-3, 5.4e-3
        xland, ztrop, pblh, bltop = rng.choice([1.0, 2.0]), r(15000, 16000), r(400, 800), r(600, 900)
    else:
        raise ValueError(family)

    psfc = 101325.0 * np.exp(-ht / 8400.0) * r(0.985, 1.01)
    eta = eta_levels(nk)
    p8w = ptop + eta * (psfc - ptop)
    p = 0.5 * (p8w[:-1] + p8w[1:])

    # temperature as a function of height above ground, iterated with the
    # hypsometric heights (two passes are plenty for a fixture)
    zw = np.zeros(nk + 1)
    for _ in range(3):
        zm = 0.5 * (zw[:-1] + zw[1:])
        t = np.where(zm < bltop, tsfc - gam_low * zm,
                     tsfc - gam_low * bltop - gam_mid * (zm - bltop))
        ttrop = tsfc - gam_low * bltop - gam_mid * (ztrop - bltop)
        t = np.where(zm > ztrop, ttrop + 1.5e-3 * (zm - ztrop), t)
        if bump is not None:
            t = t + bump[0] * np.clip((zm - bump[1]) / 400.0, 0.0, 1.0)
        t = np.maximum(t, 185.0)
        frac = np.clip(zm / 10000.0, 0.0, 1.0)
        rh = np.where(zm < bltop, rh_bl, rh_mid * (1 - 0.6 * frac))
        rh = np.where(zm > ztrop, 0.05, rh)
        es = esat(t)
        qv = np.maximum(EPS * rh * es / np.maximum(p - rh * es, 1.0), 1.0e-7)
        tv = t * (1 + 0.608 * qv)
        dz = RD * tv / G * np.log(p8w[:-1] / p8w[1:])
        zw = np.concatenate([[0.0], np.cumsum(dz)])
    zm = 0.5 * (zw[:-1] + zw[1:])
    pi = (p / 1.0e5) ** RCP
    th = t / pi

    shear = {"severe": 3.0e-3, "elevated": 2.0e-3}.get(family, 1.0e-3)
    u = r(-3, 6) + shear * zm + r(-2, 2) * np.sin(zm / 3000.0)
    v = r(-3, 3) + 0.4 * shear * zm * r(-1, 1)

    qc = np.where((rh > 0.9) & (t > 250.0), r(1e-5, 4e-4) * rh, 0.0)
    qi = np.where((rh > 0.85) & (t < 263.0), r(1e-6, 1e-4) * rh, 0.0)
    cldfra = np.clip((rh - 0.75) / 0.25, 0.0, 1.0) * r(0.4, 1.0)

    kpbl = int(np.searchsorted(zw, pblh))
    kpbl = max(1, min(kpbl, nk))
    tke = np.where(zw < pblh, r(0.3, 2.5), r(0.05, 0.2))

    hfx = {"stable": r(-60, -5), "elevated": r(-30, 20),
           "dry": r(200, 450), "deeppbl": r(250, 450)}.get(family, r(20, 350))
    qfx = {"dry": r(0, 2e-5)}.get(family, r(2e-5, 2.5e-4))
    if xland == 2.0:
        hfx = r(5, 40)
        qfx = r(5e-5, 2.0e-4)
    return dict(th=th, t=t, p=p, pi=pi, qv=qv, qc=qc, qi=qi, z=zm + ht,
                dz8w=np.diff(zw), u=u, v=v, cldfra=cldfra,
                p8w=p8w, z_at_w=zw + ht, tke_pbl=tke,
                tsk=t[0] + r(-1.0, 3.0), ust=r(0.08, 0.7), mavail=r(0.2, 1.0),
                pblh=pblh, xland=xland, ht=ht, psfc=psfc, hfx=hfx, qfx=qfx,
                tpert=r(0.0, 2.0), kpbl=kpbl)


MASS = ("th", "t", "p", "pi", "qv", "qc", "qi", "z", "dz8w", "u", "v", "cldfra")
FULL = ("p8w", "z_at_w", "tke_pbl")
SFC = ("tsk", "ust", "mavail", "pblh", "xland", "ht", "psfc", "hfx", "qfx", "tpert")


def write_case(path, cols, *, nk, nsteps, blpbl, sfclay, itime0, stepcu,
               dt, dx, cudt):
    ncol = len(cols)
    with open(path, "wb") as f:
        f.write(struct.pack("<9i", MAGIC, 1, ncol, nk, nsteps, blpbl, sfclay,
                            itime0, stepcu))
        f.write(struct.pack("<3f", dt, dx, cudt))
        for name in MASS + FULL:
            arr = np.stack([c[name] for c in cols], axis=1).astype(F32)
            f.write(np.ascontiguousarray(arr).tobytes())
        for name in SFC:
            f.write(np.array([c[name] for c in cols], dtype=F32).tobytes())
        f.write(np.array([c["kpbl"] for c in cols], dtype=np.int32).tobytes())


# (stem, seed, nk, ptop, dt, dx, cudt, stepcu, itime0, nsteps, blpbl, sfclay, families)
GROUPS = [
    ("cases-uw12km", 11, 40, 5000.0, 60.0, 12000.0, 0.0, 1, 1, 3, 9, 1,
     ["tropical"] * 6 + ["severe"] * 6 + ["elevated"] * 4 + ["marginal"] * 6
     + ["dry"] * 2 + ["terrain"] * 4 + ["cold"] * 3 + ["stable"] * 2
     + ["deeppbl"] * 3 + ["moistsat"] * 4 + ["pbltrap"] * 3),
    # cudt = 1 min at dt = 20 s: WRF's stepcu = nint(60/20) = 3, so the
    # scheme runs at itimestep 1 and 3 (cudts = 60 s != dt) and holds at 2, 4
    ("cases-myj3km", 23, 50, 5000.0, 20.0, 3000.0, 1.0, 3, 1, 4, 2, 2,
     ["tropical"] * 4 + ["severe"] * 6 + ["elevated"] * 3 + ["marginal"] * 4
     + ["dry"] * 1 + ["terrain"] * 3 + ["cold"] * 2 + ["stable"] * 2
     + ["deeppbl"] * 2 + ["moistsat"] * 1 + ["pbltrap"] * 2),
    ("cases-uw100km", 37, 30, 10000.0, 300.0, 100000.0, 0.0, 1, 2, 3, 9, 1,
     ["tropical"] * 4 + ["severe"] * 3 + ["marginal"] * 3 + ["terrain"] * 2
     + ["cold"] * 2 + ["stable"] * 1 + ["moistsat"] * 1),
    ("cases-toptrop", 41, 28, 20000.0, 30.0, 9000.0, 0.0, 1, 1, 3, 9, 1,
     ["toptrop"] * 6 + ["severe"] * 2),
]


def main(outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for (stem, seed, nk, ptop, dt, dx, cudt, stepcu, itime0, nsteps, blpbl,
         sfclay, fams) in GROUPS:
        rng = np.random.default_rng(seed)
        cols = [column(f, nk, ptop, rng) for f in fams]
        write_case(out / f"{stem}.bin", cols, nk=nk, nsteps=nsteps,
                   blpbl=blpbl, sfclay=sfclay, itime0=itime0, stepcu=stepcu,
                   dt=dt, dx=dx, cudt=cudt)
        index[stem] = dict(families=fams, nk=nk, ptop=ptop, dt=dt, dx=dx,
                           cudt=cudt, stepcu=stepcu, itimestep0=itime0,
                           nsteps=nsteps, bl_pbl_physics=blpbl,
                           sf_sfclay_physics=sfclay, seed=seed)
    (out / "cases-index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"{sum(len(g[-1]) for g in GROUPS)} columns in {len(GROUPS)} files")


if __name__ == "__main__":
    main(sys.argv[1])
