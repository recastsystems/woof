#!/usr/bin/env python3
"""Generate NTU (mp_physics=56) oracle column batches.

Writes the binary input the gfortran oracle (run_ntu.F90) and the WOOF port
runners both read.  Layout (little endian):
  int32 hdr[8] = magic 0x4E545531, ncol, nz, nsteps, ccnty, restart, 0, 0
  float32 dt
  33 mass-level fields, each (nz, ncol) float32 (level k=0 at the surface):
     TH P DZ PII ALT QV QC QR QI QS QG QH NC NR NI NS NG NH
     QDCN QTCN QCCN QRCN QNIN FI FS VI VS VG AI AS AG AH I3M
  3 w-level fields, each (nz+1, ncol): W PHB PH
  5 surface fields, each (ncol): RAINNC SNOWNC GRAPNC HAILNC XLAND
Also writes <out>.json naming every column's regime and variant.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

MASS = ("TH P DZ PII ALT QV QC QR QI QS QG QH NC NR NI NS NG NH "
        "QDCN QTCN QCCN QRCN QNIN FI FS VI VS VG AI AS AG AH I3M").split()
WLEV = ("W", "PHB", "PH")
SURF = ("RAINNC", "SNOWNC", "GRAPNC", "HAILNC", "XLAND")

G, RD, CP = 9.81, 287.0, 1004.5
RHOW, RHOI0 = 997.0, 910.0


def esat_w(t):
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def esat_i(t):
    return 611.2 * np.exp(21.8745584 * (t - 273.16) / (t - 7.66))


def grid(nz, ztop=20000.0):
    # stretched w levels: thin near the ground, ~700 m aloft
    s = np.linspace(0.0, 1.0, nz + 1)
    zw = ztop * (0.35 * s + 0.65 * s ** 2.2)
    return zw


def base_column(nz, ps, ts, lapse=0.0065, ztrop=12000.0, rh=None):
    zw = grid(nz)
    zm = 0.5 * (zw[1:] + zw[:-1])
    t = np.where(zm < ztrop, ts - lapse * zm, ts - lapse * ztrop)
    # hydrostatic pressure, layer by layer
    p = np.empty(nz)
    pw = ps
    tw = ts
    for k in range(nz):
        tm = t[k]
        dz = zm[k] - (zw[k] if k == 0 else zm[k - 1])
        pw = pw * math.exp(-G * dz / (RD * tm))
        p[k] = pw
    return zw, zm, t, p


def bump(z, z0, z1, peak=None):
    """Smooth profile, zero outside [z0, z1]."""
    x = (z - z0) / max(z1 - z0, 1.0)
    return np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 2, 0.0)


def number_from(q, d, rho):
    m = rho * math.pi / 6.0 * d ** 3
    return np.where(q > 0, q / m, 0.0)


def make_column(regime, var, nz, rng):
    scale = (0.1, 1.0, 3.0, 1.0, 0.5, 1.0, 2.0, 1.0)[var]
    numbers = ("consistent", "consistent", "consistent", "zero",
               "consistent", "perturbed", "consistent", "zero")[var]
    moments = ("consistent", "zero", "consistent", "zero",
               "perturbed", "consistent", "zero", "perturbed")[var]
    wsign = -1.0 if var in (4,) else 1.0
    ps, ts, rhbase, wpk, wz = 100000.0, 300.0, 0.8, 1.0, 5000.0
    prof = {}
    if regime == "deep_convective":
        ps, ts, rhbase, wpk, wz = 100500.0, 303.0, 0.85, 15.0, 6000.0
        prof = dict(qc=(800, 7000, 2.0e-3), qr=(0, 5000, 2.5e-3),
                    qi=(8000, 12500, 3.0e-4), qs=(6000, 11000, 6.0e-4),
                    qg=(4000, 9500, 2.0e-3), qh=(3000, 8000, 1.5e-3))
        sup = 1.003
    elif regime == "stratiform":
        ps, ts, rhbase, wpk, wz = 101000.0, 285.0, 0.9, 0.3, 5000.0
        prof = dict(qc=(1500, 3000, 1.0e-4), qr=(0, 2600, 2.0e-4),
                    qi=(7000, 10000, 5.0e-5), qs=(2500, 8000, 4.0e-4),
                    qg=(2500, 4500, 3.0e-5), qh=(0, 0, 0.0))
        sup = 1.0005
    elif regime == "warm_rain":
        ps, ts, rhbase, wpk, wz = 101200.0, 300.0, 0.92, 2.0, 2000.0
        prof = dict(qc=(500, 3000, 1.0e-3), qr=(0, 2500, 6.0e-4),
                    qi=(0, 0, 0.0), qs=(0, 0, 0.0), qg=(0, 0, 0.0),
                    qh=(0, 0, 0.0))
        sup = 1.002
    elif regime == "mixed_phase":
        ps, ts, rhbase, wpk, wz = 98000.0, 270.0, 0.9, 1.5, 2500.0
        prof = dict(qc=(800, 3500, 3.0e-4), qr=(0, 1500, 5.0e-5),
                    qi=(1500, 5000, 6.0e-5), qs=(1000, 5000, 2.5e-4),
                    qg=(800, 3500, 1.5e-4), qh=(0, 0, 0.0))
        sup = 1.001
    elif regime == "cold_snow":
        ps, ts, rhbase, wpk, wz = 101500.0, 253.0, 0.75, 0.05, 3000.0
        prof = dict(qc=(0, 0, 0.0), qr=(0, 0, 0.0),
                    qi=(2000, 6000, 8.0e-5), qs=(0, 5000, 2.0e-4),
                    qg=(0, 0, 0.0), qh=(0, 0, 0.0))
        sup = 0.97
    elif regime == "hail":
        ps, ts, rhbase, wpk, wz = 100000.0, 302.0, 0.8, 25.0, 7000.0
        prof = dict(qc=(1000, 8000, 3.0e-3), qr=(0, 6000, 3.0e-3),
                    qi=(9000, 13000, 2.0e-4), qs=(7000, 11000, 3.0e-4),
                    qg=(3000, 9000, 2.5e-3), qh=(1500, 9000, 8.0e-3))
        sup = 1.004
    elif regime == "high_terrain":
        ps, ts, rhbase, wpk, wz = 62000.0, 278.0, 0.85, 2.5, 2000.0
        prof = dict(qc=(300, 2500, 4.0e-4), qr=(0, 1200, 1.5e-4),
                    qi=(3000, 6000, 8.0e-5), qs=(500, 5000, 4.0e-4),
                    qg=(300, 3000, 4.0e-4), qh=(500, 2500, 2.0e-4))
        sup = 1.001
    else:  # edge
        ps, ts, rhbase, wpk, wz = 100000.0, 290.0, 0.5, 0.0, 3000.0
        sup = 0.6
    zw, zm, t, p = base_column(nz, ps, ts)
    t = t + rng.normal(0.0, 0.3, nz)
    es = np.where(t >= 273.15, esat_w(t), esat_i(t))
    qs_ = 0.622 * es / (p - es)
    rh = np.full(nz, rhbase) * np.exp(-zm / 9000.0)
    q = {k: np.zeros(nz) for k in ("qc", "qr", "qi", "qs", "qg", "qh")}
    for k, (z0, z1, amp) in prof.items():
        if amp > 0:
            q[k] = amp * scale * bump(zm, z0, z1) * (1.0 + 0.2 * rng.standard_normal(nz)).clip(0.3)
    cloudy = (q["qc"] + q["qi"]) > 1e-6
    rh = np.where(cloudy, sup, rh)
    qv = rh * qs_
    w_ = np.concatenate([[0.0], wsign * wpk * scale ** 0.5 * bump(zw[1:], 0, 2 * wz)])
    if regime == "edge":
        # tiny and threshold-straddling masses at every level
        tiny = np.array([0.0, 1e-15, 1e-14, 9.99e-15, 1e-12, 1e-10, 9.9e-10,
                         1e-9, 1.01e-9, 1e-8, 1e-7, 9.9e-7, 1e-6, 1.01e-6])
        for i, k in enumerate(q):
            q[k] = tiny[(np.arange(nz) + 3 * i + var) % len(tiny)]
        if var % 2 == 1:
            # ice and snow above freezing, rain below -40 C
            t = t + 25.0
        if var in (2, 3):
            t[:] = np.minimum(t, 235.0)
        if var == 5:
            qv = qv * 0.01            # strongly subsaturated
        if var == 6:
            qv = qs_ * 1.05           # strongly supersaturated, nothing in it
        w_ = np.concatenate([[0.0], (-1.0 if var % 3 == 0 else 1.0) * 3.0 * bump(zw[1:], 0, 8000)])
    rho = p / (RD * t * (1.0 + 0.61 * qv))
    alt = (1.0 + qv) / rho
    pii = (p / 1.0e5) ** (RD / CP)
    th = t / pii
    dz = np.diff(zw)
    # numbers
    dsize = dict(qc=12e-6, qr=0.8e-3, qi=60e-6, qs=1.5e-3, qg=1.2e-3,
                 qh=6e-3 if regime == "hail" else 2.5e-3)
    rhos = dict(qc=RHOW, qr=RHOW, qi=500.0, qs=100.0, qg=400.0, qh=900.0)
    n = {}
    for k in q:
        d = dsize[k] * (1.0 + 0.3 * rng.standard_normal())
        d = max(d, 0.3 * dsize[k])
        n[k] = number_from(q[k], d, rhos[k])
        if numbers == "zero":
            n[k] = np.zeros(nz)
        elif numbers == "perturbed":
            n[k] = n[k] * np.exp(rng.normal(0.0, 1.5, nz))
    v6pi = 6.0 / math.pi
    mom = {}
    rhoi = 500.0 * np.ones(nz)
    mom["VI"] = q["qi"] / rhoi
    mom["I3M"] = q["qi"] * v6pi / rhoi
    i3m0 = np.maximum(n["qi"], 1e-30) * (6e-6) ** 3
    zeta = 0.15
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mom["I3M"] > 0, mom["I3M"] / np.maximum(i3m0, 1e-38), 1.0)
        mom["FI"] = np.where(mom["I3M"] > 0, np.power(np.maximum(ratio, 1e-30), zeta) * mom["I3M"], 0.0)
        kdx = 0.65
        mom["AI"] = np.cbrt(kdx * n["qi"] * mom["I3M"] ** 2)
    s3m = q["qs"] * v6pi / rhos["qs"]
    mom["VS"] = q["qs"] / rhos["qs"]
    mom["FS"] = 0.7 * s3m
    mom["AS"] = np.cbrt(0.5 * n["qs"] * s3m ** 2)
    g3m = q["qg"] * v6pi / rhos["qg"]
    mom["VG"] = q["qg"] / rhos["qg"]
    mom["AG"] = np.cbrt(0.5 * n["qg"] * g3m ** 2)
    h3m = q["qh"] * v6pi / rhos["qh"]
    mom["AH"] = np.cbrt(0.5 * n["qh"] * h3m ** 2)
    if moments == "zero":
        for k in mom:
            mom[k] = np.zeros(nz)
    elif moments == "perturbed":
        for k in mom:
            mom[k] = mom[k] * np.exp(rng.normal(0.0, 1.0, nz))
    col = dict(TH=th, P=p, DZ=dz, PII=pii, ALT=alt, QV=qv,
               QC=q["qc"], QR=q["qr"], QI=q["qi"], QS=q["qs"], QG=q["qg"],
               QH=q["qh"], NC=n["qc"], NR=n["qr"], NI=n["qi"], NS=n["qs"],
               NG=n["qg"], NH=n["qh"], **mom)
    # aerosol defaults (used when the batch runs with restart=1)
    dry = 1.5e-9 * np.exp(-zm / 2500.0) * (1.0 + 0.1 * var)
    col["QDCN"] = dry
    col["QTCN"] = dry * 1.3
    col["QCCN"] = np.where(q["qc"] > 1e-8, 0.2 * dry, 0.0)
    col["QRCN"] = np.where(q["qr"] > 1e-8, 0.1 * dry, 0.0)
    col["QNIN"] = np.full(nz, 3.0e5) * np.exp(-zm / 5000.0)
    if regime == "edge":
        if var == 0:
            col["QDCN"][:] = 0.0          # no dry aerosol: RC_MIN branch
        if var == 7:
            col["QDCN"] = col["QTCN"] * (1.0 - 1e-8)   # all dry: RC_MAX branch
    col["W"] = w_
    col["PHB"] = G * zw
    col["PH"] = rng.normal(0.0, 2.0, nz + 1) * (zw > 0)
    col["RAINNC"] = abs(rng.normal(5.0, 3.0))
    col["SNOWNC"] = abs(rng.normal(1.0, 1.0))
    col["GRAPNC"] = abs(rng.normal(0.5, 0.5))
    col["HAILNC"] = abs(rng.normal(0.1, 0.1))
    col["XLAND"] = 1.0 if var % 2 == 0 else 2.0
    return col


REGIMES = ("deep_convective", "stratiform", "warm_rain", "mixed_phase",
           "cold_snow", "hail", "high_terrain", "edge")


def write_batch(path, cols, nz, nsteps, ccnty, restart, dt):
    ncol = len(cols)
    with open(path, "wb") as f:
        np.array([0x4E545531, ncol, nz, nsteps, ccnty, restart, 0, 0],
                 dtype="<i4").tofile(f)
        np.array([dt], dtype="<f4").tofile(f)
        for name in MASS:
            a = np.stack([c[name] for c in cols], axis=1).astype("<f4")
            assert a.shape == (nz, ncol), (name, a.shape)
            a.tofile(f)
        for name in WLEV:
            a = np.stack([c[name] for c in cols], axis=1).astype("<f4")
            assert a.shape == (nz + 1, ncol)
            a.tofile(f)
        for name in SURF:
            np.array([c[name] for c in cols], dtype="<f4").tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--nz", type=int, default=40)
    args = ap.parse_args()
    batches = [
        # name, regimes x variants, ccnty, restart, dt, steps, seed
        ("A", [(r, v) for r in REGIMES for v in range(8)], 2, 0, 20.0, 3, 1),
        ("B", [(r, v) for r in REGIMES for v in range(8)], 2, 1, 60.0, 3, 2),
        ("C", [(r, v) for r in REGIMES[:4] for v in (1, 2)], 1, 0, 12.0, 2, 3),
        ("D", [(r, v) for r in REGIMES[4:] for v in (1, 2)], 3, 0, 30.0, 2, 4),
        ("E", [(r, v) for r in REGIMES for v in (2,)], 4, 0, 20.0, 2, 5),
    ]
    meta = {}
    for name, members, ccnty, restart, dt, steps, seed in batches:
        rng = np.random.default_rng(seed)
        cols = [make_column(r, v, args.nz, rng) for r, v in members]
        path = f"{args.outdir}/ntu_batch_{name}.bin"
        write_batch(path, cols, args.nz, steps, ccnty, restart, dt)
        meta[name] = dict(columns=[f"{r}/v{v}" for r, v in members],
                          ccnty=ccnty, restart=restart, dt=dt, steps=steps,
                          nz=args.nz, seed=seed)
    with open(f"{args.outdir}/ntu_batches.json", "w") as f:
        json.dump(meta, f, indent=1)
    print({k: len(v["columns"]) for k, v in meta.items()})


if __name__ == "__main__":
    main()
