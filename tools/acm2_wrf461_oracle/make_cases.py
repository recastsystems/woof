"""Write the ACM2 PBL oracle's column fixtures (inputs only).

Each fixture is one WRF tile row of independent columns, built to reach the
scheme's branches rather than to be a climatology:

* convective daytime land at several mixed-layer depths and heat fluxes,
  with and without a superadiabatic first layer (``THETAV(1) > THETAV(2)``
  is one of the four conditions of the free-convection arm, so both sides
  of ``NOCONV`` are reached at the same depth and flux);
* shallow convective layers whose PBL top sits at or below level 3
  (``KLPBL <= 3``: the local-only arm under an unstable surface);
* stable nights from weak to strong inversions, with and without a
  low-level jet;
* neutral columns, including exactly zero surface fluxes (``TSTV`` is
  floored to ``SIGN(1.0E-6, 0.0)``) and near-zero ones on both sides;
* sea columns (large latent flux, small sensible flux, both signs);
* cloud-topped and ice-topped columns (the moist Richardson adjustment of
  EDDYX fires where ``qc+qi > 0.01E-3``);
* weak-wind columns (the ``WSSQ`` floor of 0.1) and strong-shear ones;
* first-layer thicknesses from 12 m to 65 m, so the 30 m source-layer
  search ends at level 1, 2 or 3 (``KSRC``).

``XTIME`` is WRF's ``itimestep``: step 1 never takes the free-convection
arm and never adds the convective temperature excess, every later step
may, so every fixture starts at its own ``xtime0`` and the oracle runs
several consecutive steps.

Usage: python make_cases.py OUTDIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

F = np.float32

# WRF module_model_constants, evaluated in float32 in WRF's own order.
G = F(9.81)
R_D = F(287.0)
R_V = F(461.6)
CP = F(F(7.0) * R_D) / F(2.0)
RCP = R_D / CP
EP_1 = F(R_V / R_D) - F(1.0)
P1000 = 1.0e5


def grid(nz: int, dz1: float, ztop: float) -> np.ndarray:
    """Interface heights (nz+1,) from 0 with first layer dz1 to ztop."""
    lo, hi = 1.0, 1.5
    for _ in range(200):
        r = 0.5 * (lo + hi)
        dz = np.minimum(dz1 * r ** np.arange(nz), 900.0)
        if dz.sum() < ztop:
            lo = r
        else:
            hi = r
    dz = np.minimum(dz1 * r ** np.arange(nz), 900.0)
    dz *= ztop / dz.sum()
    dz[0] = dz1
    return np.concatenate([[0.0], np.cumsum(dz)])


def theta_profile(z, kind, p):
    th_s = p["th_s"]
    trop = 12000.0
    gam = p.get("gam", 4.0e-3)
    above = np.where(z > trop, gam * (trop) + 0.02 * (z - trop), gam * z)
    if kind == "conv":
        h = p["h"]
        dsl = p.get("dsl", 1.0)          # superadiabatic surface-layer drop
        dth = p.get("dth", 2.0)          # inversion jump
        zi = p.get("dzi", 150.0)
        ml = th_s - dsl * np.exp(-z / 40.0)
        jump = dth * np.clip((z - h) / zi + 0.5, 0.0, 1.0)
        free = np.where(z > h, above - gam * h, 0.0)
        return ml + jump + free
    if kind == "stable":
        hs = p.get("hs", 150.0)
        dth = p["dth"]
        return th_s + dth * (1.0 - np.exp(-z / hs)) + above
    if kind == "neutral":
        h = p.get("h", 800.0)
        free = np.where(z > h, above - gam * h, 0.0)
        return th_s + free + p.get("tiny", 0.0) * z
    if kind == "allunstable":
        return th_s - 0.002 * z
    raise ValueError(kind)


def column(nz, zf, kind, p, rng):
    za = 0.5 * (zf[1:] + zf[:-1])
    dz = np.diff(zf)
    th = theta_profile(za, kind, p)
    h = p.get("h", 300.0)
    qs = p.get("qv_s", 0.008)
    qv = np.where(za < h, qs, qs * np.exp(-(za - h) / 2500.0))
    qv = np.where(za > 12000.0, 3.0e-6, qv)
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    for zb, zt, qmax in p.get("qc_layers", ()):
        qc = np.where((za > zb) & (za < zt), qmax, qc)
    for zb, zt, qmax in p.get("qi_layers", ()):
        qi = np.where((za > zb) & (za < zt), qmax, qi)
    # hydrostatic pressure from the surface, float64
    psfc = p.get("psfc", 98000.0)
    pint = np.empty(nz + 1)
    pint[0] = psfc
    pmid = np.empty(nz)
    for k in range(nz):
        pg = pint[k]
        for _ in range(3):
            pm = pint[k] * np.exp(-0.5 * 9.81 * dz[k] / (287.0 * th[k] * (pg / P1000) ** (287.0 / 1004.5) * (1 + 0.61 * qv[k])))
            pg = pm
        pmid[k] = pm
        tv = th[k] * (pm / P1000) ** (287.0 / 1004.5) * (1 + 0.61 * qv[k])
        pint[k + 1] = pint[k] * np.exp(-9.81 * dz[k] / (287.0 * tv))
    pi = (pmid / P1000) ** (287.0 / 1004.5)
    rho = pmid / (287.0 * th * pi * (1 + 0.61 * qv)) * (1 + qv)
    # winds
    ug = p.get("ug", 8.0)
    z0 = p.get("z0", 0.1)
    veer = np.deg2rad(p.get("veer", 25.0))
    prof = np.clip(np.log(za / z0) / np.log(max(h, 50.0) / z0), 0.05, 1.0)
    speed = ug * prof + p.get("shear", 0.0) * np.maximum(za - h, 0.0)
    jet = p.get("jet")
    if jet:
        zj, uj = jet
        speed = speed + uj * np.exp(-((za - zj) / (0.6 * zj)) ** 2)
    ang = p.get("dir", 0.0) + veer * np.clip(za / max(h, 50.0), 0.0, 1.0)
    u = speed * np.cos(ang) + rng.normal(0.0, 0.05, nz)
    v = speed * np.sin(ang) + rng.normal(0.0, 0.05, nz)
    th = th + rng.normal(0.0, 0.01, nz) * (za > 200.0)
    return dict(u=u, v=v, p=pmid, dz8w=dz, th=th, qv=qv, qc=qc, qi=qi,
                rho=rho, pi=pi, psfc=psfc)


def families(rng):
    """(family, kind, params, surface) in a fixed order."""
    out = []
    for h in (300.0, 800.0, 1500.0, 2500.0, 3500.0):
        for hfx in (80.0, 250.0, 450.0):
            out.append(("conv-land", "conv",
                        dict(th_s=300.0 + h / 1000.0, h=h, dsl=1.5,
                             dth=2.5, ug=6.0, qv_s=0.010),
                        dict(hfx=hfx, qfx=hfx / 2.5e6 * 0.6,
                             ust=0.25 + hfx / 2000.0)))
    # same depths, no superadiabatic first layer: NOCONV = 0 although the
    # surface is unstable (THETAV(1) <= THETAV(2))
    for h in (800.0, 1500.0, 2500.0):
        out.append(("conv-no-sal", "conv",
                    dict(th_s=301.0, h=h, dsl=-0.2, dth=2.0, ug=7.0),
                    dict(hfx=200.0, qfx=6.0e-5, ust=0.35)))
    # shallow convective layers, KLPBL <= 3
    for h, hfx in ((60.0, 30.0), (90.0, 60.0), (120.0, 15.0), (150.0, 40.0)):
        out.append(("conv-shallow", "conv",
                    dict(th_s=290.0, h=h, dsl=0.6, dth=4.0, dzi=40.0, ug=4.0),
                    dict(hfx=hfx, qfx=1.5e-5, ust=0.2)))
    # stable nights
    for dth in (2.0, 6.0, 12.0):
        for ust in (0.05, 0.15, 0.32):
            out.append(("stable-night", "stable",
                        dict(th_s=285.0, dth=dth, hs=120.0, ug=5.0 + 20 * ust,
                             h=150.0, qv_s=0.006),
                        dict(hfx=-15.0 - 4.0 * dth * ust * 10.0,
                             qfx=-2.0e-6, ust=ust)))
    for zj, uj in ((250.0, 12.0), (400.0, 18.0)):
        out.append(("stable-llj", "stable",
                    dict(th_s=290.0, dth=8.0, hs=200.0, ug=4.0, h=200.0,
                         jet=(zj, uj)),
                    dict(hfx=-25.0, qfx=0.0, ust=0.2)))
    # neutral, including exactly zero fluxes
    for hfx, qfx in ((0.0, 0.0), (0.0, 0.0), (0.5, 0.0), (-0.5, 0.0),
                     (0.0, 1.0e-6), (2.0, 5.0e-6)):
        out.append(("neutral", "neutral",
                    dict(th_s=288.0, h=900.0, ug=14.0, gam=5.0e-3),
                    dict(hfx=hfx, qfx=qfx, ust=0.55)))
    # sea
    for hfx, qfx, ust in ((5.0, 1.2e-4, 0.25), (25.0, 2.5e-4, 0.4),
                          (60.0, 3.0e-4, 0.6), (-8.0, 2.0e-5, 0.2),
                          (12.0, 8.0e-5, 0.15), (40.0, 2.0e-4, 0.5),
                          (-2.0, 5.0e-6, 0.08), (90.0, 4.0e-4, 0.8)):
        h = 600.0 if hfx > 0 else 200.0
        kind = "conv" if hfx > 0 else "stable"
        out.append(("sea", kind,
                    dict(th_s=293.0, h=h, dsl=0.4, dth=1.5, hs=100.0,
                         ug=9.0, z0=2.0e-4, qv_s=0.014),
                    dict(hfx=hfx, qfx=qfx, ust=ust)))
    # cloud- and ice-topped
    clouds = (
        ("stratocumulus", "conv", dict(th_s=289.0, h=900.0, dsl=0.3,
                                       dth=6.0, dzi=80.0, ug=7.0,
                                       qv_s=0.009,
                                       qc_layers=[(600.0, 950.0, 3.0e-4)]),
         dict(hfx=20.0, qfx=4.0e-5, ust=0.25)),
        ("cumulus-topped", "conv", dict(th_s=303.0, h=1800.0, dsl=1.5,
                                        dth=1.0, ug=5.0, qv_s=0.016,
                                        qc_layers=[(1700.0, 3200.0, 6.0e-4)],
                                        qi_layers=[(6000.0, 9000.0, 5.0e-5)]),
         dict(hfx=300.0, qfx=1.2e-4, ust=0.4)),
        ("cirrus-only", "stable", dict(th_s=284.0, dth=5.0, ug=8.0, h=150.0,
                                       qi_layers=[(8000.0, 11000.0, 2.0e-5)]),
         dict(hfx=-20.0, qfx=0.0, ust=0.15)),
        ("fog", "stable", dict(th_s=282.0, dth=3.0, hs=60.0, ug=2.0,
                               h=80.0, qv_s=0.0075,
                               qc_layers=[(0.0, 120.0, 1.5e-4)]),
         dict(hfx=-5.0, qfx=-1.0e-6, ust=0.06)),
        ("mixed-phase", "conv", dict(th_s=275.0, h=1200.0, dsl=0.8,
                                     dth=3.0, ug=10.0, qv_s=0.004,
                                     qc_layers=[(900.0, 1300.0, 1.0e-4)],
                                     qi_layers=[(1100.0, 1500.0, 4.0e-5)]),
         dict(hfx=60.0, qfx=2.0e-5, ust=0.35)),
        ("threshold-cloud", "conv", dict(th_s=296.0, h=1000.0, dsl=1.0,
                                         dth=2.0, ug=6.0,
                                         qc_layers=[(500.0, 900.0, 1.0e-5)],
                                         qi_layers=[(500.0, 900.0, 1.0e-9)]),
         dict(hfx=150.0, qfx=5.0e-5, ust=0.3)),
    )
    for name, kind, prm, sfc in clouds:
        out.append(("cloud-" + name, kind, prm, sfc))
    # weak wind and strong shear
    for ug, ust, hfx in ((0.3, 0.02, 40.0), (0.2, 0.01, -3.0),
                         (0.5, 0.05, 150.0), (0.1, 0.004, 5.0)):
        out.append(("weak-wind", "conv" if hfx > 0 else "stable",
                    dict(th_s=298.0, h=700.0, dsl=0.8, dth=2.0, dthst=3.0,
                         dth_s=3.0, ug=ug, hs=100.0, veer=90.0)
                    | ({} if hfx > 0 else {"dth": 3.0}),
                    dict(hfx=hfx, qfx=1.0e-5, ust=ust)))
    for shear in (0.004, 0.012, 0.02):
        out.append(("strong-shear", "conv",
                    dict(th_s=297.0, h=1000.0, dsl=0.7, dth=1.5, ug=12.0,
                         shear=shear),
                    dict(hfx=120.0, qfx=4.0e-5, ust=0.5)))
    # a deep convective PBL reaching toward the upper troposphere
    for h in (5000.0, 6500.0):
        out.append(("deep", "conv",
                    dict(th_s=310.0, h=h, dsl=2.5, dth=1.0, ug=8.0,
                         qv_s=0.006, gam=6.0e-3),
                    dict(hfx=550.0, qfx=8.0e-5, ust=0.6)))
    # seeded draws across the whole regime space
    for _ in range(12):
        hfx = float(rng.uniform(-60.0, 400.0))
        h = float(rng.uniform(100.0, 3000.0))
        out.append(("drawn", "conv" if hfx > 0 else "stable",
                    dict(th_s=float(rng.uniform(280.0, 305.0)), h=h,
                         dsl=float(rng.uniform(-0.3, 2.0)),
                         dth=float(rng.uniform(0.5, 8.0)),
                         hs=float(rng.uniform(60.0, 300.0)),
                         ug=float(rng.uniform(1.0, 15.0)),
                         veer=float(rng.uniform(-30.0, 60.0)),
                         qv_s=float(rng.uniform(0.002, 0.016))),
                    dict(hfx=hfx, qfx=float(rng.uniform(-5e-6, 2e-4)),
                         ust=float(rng.uniform(0.05, 0.7)))))
    return out


def build(nz, dt, xtime0, nsteps, ztop, dz1_choices, seed, outdir, name):
    rng = np.random.default_rng(seed)
    cases = families(rng)
    ncol = len(cases)
    f3 = {k: np.zeros((ncol, nz)) for k in
          ("u", "v", "p", "dz8w", "th", "t", "qv", "qc", "qi", "rho", "pi")}
    f2 = {k: np.zeros(ncol) for k in
          ("ust", "hfx", "qfx", "tsk", "psfc", "wspd", "psim", "gz1oz0",
           "mut", "pblh", "regime", "rmol")}
    index = []
    for i, (family, kind, prm, sfc) in enumerate(cases):
        dz1 = dz1_choices[i % len(dz1_choices)]
        zf = grid(nz, dz1, ztop)
        col = column(nz, zf, kind, prm, rng)
        for k in ("u", "v", "p", "dz8w", "th", "qv", "qc", "qi", "rho", "pi"):
            f3[k][i] = col[k]
        th32 = col["th"].astype(F)
        pi32 = col["pi"].astype(F)
        f3["t"][i] = th32 * pi32
        ws1 = float(np.hypot(col["u"][0], col["v"][0]))
        hfx = sfc["hfx"]
        wc = 0.0 if hfx <= 0 else 1.2 * (9.81 / 300.0 * hfx / 1200.0
                                         * prm.get("h", 300.0)) ** (1 / 3)
        f2["ust"][i] = sfc["ust"]
        f2["hfx"][i] = hfx
        f2["qfx"][i] = sfc["qfx"]
        f2["tsk"][i] = prm["th_s"] * col["pi"][0] + (2.0 if hfx > 0 else -2.0)
        f2["psfc"][i] = col["psfc"]
        f2["wspd"][i] = max(np.hypot(ws1, wc), 0.1)
        f2["psim"][i] = -0.5 if hfx > 0 else 1.0
        f2["gz1oz0"][i] = np.log(0.5 * dz1 / prm.get("z0", 0.1))
        f2["mut"][i] = col["psfc"] - 5000.0
        # carried PBLH: a cold start (0) on every third column, else a
        # plausible previous value; regime/rmol as a surface layer leaves
        # them (the scheme overwrites RMOL and REGIME's convective cells).
        f2["pblh"][i] = 0.0 if i % 3 == 0 else prm.get("h", 300.0)
        f2["regime"][i] = 4.0 if hfx > 0 else 1.0
        f2["rmol"][i] = -0.01 if hfx > 0 else 0.02
        index.append({"column": i, "family": family, "kind": kind,
                      "dz1_m": dz1, "hfx": hfx, "qfx": sfc["qfx"],
                      "ust": sfc["ust"],
                      "pbl_depth_m": prm.get("h"),
                      "cloud": bool(prm.get("qc_layers") or
                                    prm.get("qi_layers"))})
    header = np.array([ncol, nz, nsteps, xtime0], dtype="<i4")
    consts = np.array([dt, EP_1, G, RCP, R_D, CP], dtype="<f4")
    path = Path(outdir) / f"{name}.in.bin"
    with open(path, "wb") as fh:
        fh.write(header.tobytes())
        fh.write(consts.tobytes())
        for k in ("u", "v", "p", "dz8w", "th", "t", "qv", "qc", "qi", "rho",
                  "pi"):
            fh.write(f3[k].astype("<f4").tobytes())
        for k in ("ust", "hfx", "qfx", "tsk", "psfc", "wspd", "psim",
                  "gz1oz0", "mut", "pblh", "regime", "rmol"):
            fh.write(f2[k].astype("<f4").tobytes())
    (Path(outdir) / f"{name}.index.json").write_text(json.dumps(
        {"name": name, "ncol": ncol, "nz": nz, "dtpbl": dt,
         "xtime0": xtime0, "nsteps": nsteps, "ztop_m": ztop, "seed": seed,
         "columns": index}, indent=1) + "\n")
    return ncol


def build_fatal(outdir):
    """Two columns: an ordinary convective one and a column whose virtual
    potential temperature falls all the way to the model top, so the bulk
    Richardson number never reaches RIC and WRF stops
    (module_bl_acm.F:464-470).  Column 2 is the one WRF names."""
    nz = 30
    rng = np.random.default_rng(7)
    zf = grid(nz, 30.0, 15000.0)
    cols = [column(nz, zf, "conv", dict(th_s=300.0, h=1000.0, dsl=1.0,
                                          dth=2.0, ug=6.0), rng),
            column(nz, zf, "allunstable", dict(th_s=300.0, ug=6.0, h=500.0),
                   rng)]
    header = np.array([2, nz, 2, 1], dtype="<i4")
    consts = np.array([60.0, EP_1, G, RCP, R_D, CP], dtype="<f4")
    with open(Path(outdir) / "fatal.in.bin", "wb") as fh:
        fh.write(header.tobytes())
        fh.write(consts.tobytes())
        for k in ("u", "v", "p", "dz8w", "th", "t", "qv", "qc", "qi", "rho",
                  "pi"):
            if k == "t":
                arr = np.stack([c["th"].astype(F) * c["pi"].astype(F)
                                for c in cols])
            else:
                arr = np.stack([c[k] for c in cols])
            fh.write(arr.astype("<f4").tobytes())
        vals = dict(ust=[0.3, 0.3], hfx=[150.0, 150.0], qfx=[3e-5, 3e-5],
                    tsk=[302.0, 302.0], psfc=[c["psfc"] for c in cols],
                    wspd=[5.0, 5.0], psim=[-0.5, -0.5], gz1oz0=[7.0, 7.0],
                    mut=[93000.0, 93000.0], pblh=[1000.0, 1000.0],
                    regime=[4.0, 4.0], rmol=[-0.01, -0.01])
        for k in ("ust", "hfx", "qfx", "tsk", "psfc", "wspd", "psim",
                  "gz1oz0", "mut", "pblh", "regime", "rmol"):
            fh.write(np.asarray(vals[k], dtype="<f4").tobytes())


def main():
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    # A: 40 levels, a 60 s PBL step from the forecast's first step.
    build(40, 60.0, 1, 6, 18000.0, (12.0, 20.0, 28.0, 35.0, 50.0, 65.0),
          2026100801, outdir, "acm2-l40")
    # B: 60 levels, a 180 s PBL step late in a forecast (more sub-steps).
    build(60, 180.0, 41, 4, 20000.0, (35.0, 12.0, 65.0, 20.0, 28.0, 50.0),
          2026100802, outdir, "acm2-l60")
    build_fatal(outdir)


if __name__ == "__main__":
    main()
