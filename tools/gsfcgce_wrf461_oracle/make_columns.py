#!/usr/bin/env python3
"""Column fixtures for the WRF v4.6.1 Goddard GCE oracle.

    python make_columns.py <outdir>

Writes ``<outdir>/<set>.bin`` and ``<outdir>/index.json``.  Two sets of the
same 66 columns (40 levels): ``main`` runs 12 steps of 12 s, ``longdt``
runs 6 steps of 60 s (deep sedimentation sub-stepping).  Every case is run
under all eight switch combinations ``gsfcgce_hail`` in {0, 1} x
``gsfcgce_2ice`` in {0, 1, 2, 3}.

Families (labels in index.json): deep convective, hail-producing,
stratiform ice with a melting layer, warm shallow, dry, and edge cases
(exact zeros, the 1e-8 sedimentation threshold, cmin1/cmin2 thresholds,
tiny negatives, subnormals, the bnd3 autoconversion threshold, a level at
exactly T = t0 and T = t00, the 253.16-258.16 K overlap band, ultracold
air, hydrometeor in the never-sedimenting top level, thin surface layers).

Inputs are deterministic (fixed seed) float32 in the layout run_gsfcgce.F90
reads: twelve (ncol, nk) Fortran-order fields, then four (ncol) fields.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

NK = 40
F32 = np.float32
RD, CP, G = 287.0, 1004.0, 9.81
IN3D = ["th", "qv", "qc", "qr", "qi", "qs", "qg", "rho", "pii", "p", "z",
        "dz8w"]
IN2D = ["ht", "rainnc", "snownc", "graupelnc"]
COMBOS = [(h, i) for i in (0, 1, 2, 3) for h in (0, 1)]


def qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def bump(h, amp, hc, w):
    return amp * np.exp(-((h - hc) / w) ** 2)


class Column:
    def __init__(self, label, rng, *, ht=0.0, ts=300.0, lapse=6.5e-3,
                 htrop=12000.0, rh=0.8, top=20000.0, thin=False):
        self.label = label
        s = np.linspace(0.0, 1.0, NK + 1)
        hw = top * (0.62 * s ** 2 + 0.38 * s)
        if thin:
            hw[1] = 12.0
            hw[2] = 30.0
        z8w = (F32(ht) + hw.astype(F32)).astype(F32)
        self.ht = F32(ht)
        self.dz8w = (z8w[1:] - z8w[:-1]).astype(F32)
        self.z = (F32(0.5) * (z8w[:-1] + z8w[1:])).astype(F32)
        h = self.z.astype(np.float64) - ht
        t = np.where(h < htrop, ts - lapse * h, ts - lapse * htrop)
        ps = 101325.0 * np.exp(-G * ht / (RD * ts))
        lnp = np.log(ps) - np.cumsum(G / (RD * t) * np.diff(
            np.concatenate([[0.0], h])))
        p = np.exp(lnp)
        rhv = np.broadcast_to(np.asarray(rh, dtype=np.float64), h.shape)
        qv = rhv * qsat(t, p)
        self.h = h
        self.t = t
        self.p = p.astype(F32)
        self.qv = qv.astype(F32)
        pii = (self.p.astype(np.float64) / 1.0e5) ** (RD / CP)
        self.pii = pii.astype(F32)
        self.th = (t / self.pii.astype(np.float64)).astype(F32)
        self.rho = (p / (RD * t * (1.0 + 0.608 * qv))).astype(F32)
        self.q = {n: np.zeros(NK, dtype=F32) for n in
                  ("qc", "qr", "qi", "qs", "qg")}
        self.rainnc = F32(rng.uniform(0.0, 5.0))
        self.snownc = F32(rng.uniform(0.0, 1.0))
        self.graupelnc = F32(rng.uniform(0.0, 0.5))

    def add(self, name, amp, hc, w, *, tmax=None, tmin=None):
        v = bump(self.h, amp, hc, w)
        if tmax is not None:
            v = np.where(self.t <= tmax, v, 0.0)
        if tmin is not None:
            v = np.where(self.t >= tmin, v, 0.0)
        self.q[name] = (self.q[name].astype(np.float64) + v).astype(F32)

    def fields(self):
        out = dict(th=self.th, qv=self.qv, rho=self.rho, pii=self.pii,
                   p=self.p, z=self.z, dz8w=self.dz8w, **self.q)
        return {k: np.asarray(v, dtype=F32) for k, v in out.items()}


def make_columns(rng):
    cols = []
    u = rng.uniform
    for n in range(12):     # deep convective
        c = Column(f"deep-convective-{n}", rng, ht=u(0, 600), ts=u(296, 306),
                   rh=np.clip(u(0.85, 1.0) + 0.04 * np.sin(np.arange(NK)), 0, 1.03))
        c.add("qc", u(5e-4, 3e-3), u(2500, 5500), u(1500, 3000))
        c.add("qr", u(1e-3, 8e-3), u(1000, 4000), u(1500, 3000))
        c.add("qi", u(5e-5, 1e-3), u(9000, 12000), u(1500, 3000), tmax=268)
        c.add("qs", u(3e-4, 2e-3), u(7000, 10000), u(1500, 2500), tmax=272)
        c.add("qg", u(1e-3, 6e-3), u(4500, 8000), u(1500, 2500))
        cols.append(c)
    for n in range(8):      # hail-producing
        c = Column(f"hail-producing-{n}", rng, ht=u(200, 1500), ts=u(300, 308),
                   lapse=u(7.0e-3, 8.0e-3), htrop=u(13000, 15000), rh=0.92)
        c.add("qg", u(5e-3, 1.5e-2), u(3000, 7000), u(1500, 3000))
        c.add("qr", u(3e-3, 1e-2), u(1500, 3500), u(1000, 2000))
        c.add("qc", u(1e-3, 4e-3), u(4000, 7000), u(2000, 3000))
        c.add("qi", u(1e-4, 2e-3), u(10000, 13000), u(1500, 3000), tmax=260)
        c.add("qs", u(1e-4, 1e-3), u(8000, 11000), u(1500, 2500), tmax=270)
        cols.append(c)
    for n in range(10):     # stratiform ice with a melting layer
        c = Column(f"stratiform-ice-{n}", rng, ts=u(278, 286),
                   lapse=u(5.5e-3, 6.5e-3), rh=np.where(
                       np.arange(NK) > 8, u(0.9, 1.0), u(0.75, 0.95)))
        c.add("qs", u(1e-4, 1e-3), u(3500, 7000), u(1500, 3000), tmax=275)
        c.add("qi", u(1e-5, 3e-4), u(6000, 9000), u(1500, 3000), tmax=265)
        c.add("qr", u(1e-5, 6e-4), u(0, 1500), u(800, 1500))
        c.add("qc", u(0, 3e-4), u(1500, 3500), u(500, 1500))
        c.add("qg", u(0, 3e-4), u(2500, 4500), u(800, 1500))
        cols.append(c)
    for n in range(8):      # warm shallow
        c = Column(f"warm-shallow-{n}", rng, ts=u(295, 302), rh=np.where(
            np.arange(NK) < 12, u(0.97, 1.02), 0.5))
        c.add("qc", u(5e-4, 2.6e-3), u(800, 2500), u(400, 1000))
        c.add("qr", u(0, 1e-3), u(300, 1500), u(400, 1000))
        cols.append(c)
    for n in range(6):      # dry, evaporating or empty
        c = Column(f"dry-{n}", rng, ht=u(0, 2000), ts=u(285, 310),
                   rh=u(0.15, 0.5))
        if n % 2:
            c.add("qr", u(1e-5, 5e-4), u(1000, 3000), u(500, 1000))
            c.add("qs", u(1e-5, 2e-4), u(4000, 6000), u(500, 1000))
            c.add("qg", u(1e-5, 2e-4), u(3000, 5000), u(500, 1000))
        cols.append(c)
    cols.extend(edge_columns(rng))
    return cols


def edge_columns(rng):
    u = rng.uniform
    cols = []

    def base(label, **kw):
        c = Column(label, rng, ts=kw.pop("ts", 293.0), **kw)
        return c

    c = base("edge-all-zero", rh=0.0)
    c.qv[:] = F32(0.0)
    cols.append(c)
    c = base("edge-sed-threshold", rh=0.6)
    for name in ("qr", "qs", "qg", "qi"):
        c.q[name][5] = F32(1.0e-8)
        c.q[name][6] = np.nextafter(F32(1.0e-8), F32(1.0))
        c.q[name][7] = np.nextafter(F32(1.0e-8), F32(0.0))
        c.q[name][30] = F32(2.0e-8)
    cols.append(c)
    c = base("edge-cmin-thresholds", rh=0.9)
    for name in ("qc", "qr", "qi", "qs", "qg"):
        c.q[name][3] = F32(1.0e-20)
        c.q[name][4] = np.nextafter(F32(1.0e-20), F32(1.0))
        c.q[name][10] = F32(1.0e-12)
        c.q[name][11] = np.nextafter(F32(1.0e-12), F32(1.0))
        c.q[name][25] = F32(1.0e-19)
        c.q[name][26] = np.nextafter(F32(1.0e-19), F32(1.0))
    cols.append(c)
    c = base("edge-tiny-negative", rh=0.95)
    for name in ("qc", "qr", "qi", "qs", "qg"):
        c.q[name][2] = F32(-1.0e-12)
        c.q[name][20] = F32(-3.0e-9)
        c.q[name][28] = F32(-1.0e-7)
    c.add("qr", 2e-4, 2000, 1000)
    cols.append(c)
    c = base("edge-subnormal", rh=0.9)
    for name in ("qc", "qr", "qi", "qs", "qg"):
        c.q[name][1] = F32(1.0e-40)
        c.q[name][15] = F32(1.4e-45)
        c.q[name][27] = F32(1.17e-38)
    cols.append(c)
    c = base("edge-bnd3-autoconversion", rh=1.0)
    c.q["qc"][2:8] = np.array([2.0e-3, np.nextafter(F32(2.0e-3), F32(1.0)),
                               np.nextafter(F32(2.0e-3), F32(0.0)), 2.5e-3,
                               1.0e-4, 9.9e-5], dtype=F32)
    cols.append(c)
    c = base("edge-exact-t0-t00", ts=280.0, rh=1.0)
    c.pii[4] = F32(1.0)
    c.th[4] = F32(273.16)
    c.pii[22] = F32(1.0)
    c.th[22] = F32(238.16)
    c.q["qc"][4] = F32(3e-4)
    c.q["qi"][4] = F32(2e-4)
    c.q["qs"][4] = F32(2e-4)
    c.q["qg"][4] = F32(2e-4)
    c.q["qc"][22] = F32(2e-4)
    c.q["qi"][22] = F32(1e-4)
    cols.append(c)
    c = base("edge-overlap-band", ts=268.0, lapse=1.0e-3, rh=1.02)
    c.add("qc", 4e-4, 3000, 3000)
    c.add("qi", 1e-4, 3000, 3000)
    cols.append(c)
    c = base("edge-ultracold", ts=230.0, lapse=4.0e-3, htrop=5000.0, rh=1.05)
    c.add("qi", 5e-4, 4000, 3000)
    c.add("qs", 5e-4, 4000, 3000)
    c.add("qg", 2e-4, 4000, 3000)
    c.add("qc", 1e-4, 1000, 800)
    cols.append(c)
    c = base("edge-top-level", ts=300.0, rh=0.7)
    for name in ("qc", "qr", "qi", "qs", "qg"):
        c.q[name][NK - 1] = F32(5e-4)
        c.q[name][NK - 2] = F32(4e-4)
    cols.append(c)
    for n in range(2):
        c = base(f"edge-thin-surface-layer-{n}", ts=298.0, rh=0.95, thin=True)
        c.add("qr", u(2e-3, 6e-3), 0, 2500)
        c.add("qg", u(1e-3, 4e-3), 1500, 2500)
        c.add("qs", u(1e-4, 1e-3), 4000, 2500)
        c.add("qi", u(1e-4, 1e-3), 9000, 2500)
        cols.append(c)
    c = base("edge-supersaturated-ice", ts=262.0, lapse=5e-3, rh=1.0)
    c.qv[:] = (c.qv.astype(np.float64) * 1.08).astype(F32)
    c.add("qi", 2e-5, 3000, 2000)
    cols.append(c)
    c = base("edge-wet-growth-cold", ts=292.0, lapse=8.5e-3, htrop=15000,
             rh=0.95)
    c.add("qg", 8e-3, 7000, 2500)
    c.add("qc", 3e-3, 7000, 2500)
    c.add("qi", 2e-3, 7000, 2500)
    c.add("qs", 2e-3, 7000, 2500)
    cols.append(c)
    c = base("edge-snow-dominant-rimed", ts=271.0, lapse=5e-3, rh=1.0)
    c.add("qs", 3e-3, 1500, 2000)
    c.add("qc", 1e-3, 1500, 2000)
    c.add("qr", 5e-5, 500, 500)
    cols.append(c)
    c = base("edge-rain-below-1e-4-cold", ts=270.0, lapse=5e-3, rh=0.98)
    c.add("qr", 9e-5, 1000, 1500)
    c.add("qi", 2e-4, 3000, 1500)
    c.add("qs", 5e-5, 3000, 1500)
    c.add("qc", 2e-4, 2000, 1500)
    cols.append(c)
    c = base("edge-melting-graupel-warm", ts=303.0, rh=0.6)
    c.add("qg", 6e-3, 2500, 1500)
    c.add("qs", 2e-3, 2500, 1500)
    c.add("qi", 5e-4, 2500, 1500)
    cols.append(c)
    c = base("edge-huge-rain", ts=301.0, rh=0.99)
    c.add("qr", 2.5e-2, 1500, 1500)
    c.add("qg", 2.0e-2, 4000, 1500)
    cols.append(c)
    c = base("edge-high-terrain", ht=3200.0, ts=285.0, rh=0.9)
    c.add("qs", 1e-3, 2000, 1500)
    c.add("qr", 5e-4, 300, 500)
    cols.append(c)
    c = base("edge-pidw-all-bins", ts=275.0, lapse=6.5e-3, rh=1.0)
    c.add("qc", 6e-4, 3000, 3500)
    c.add("qi", 1e-4, 3000, 3500)
    cols.append(c)
    c = base("edge-qv-zero-with-ice", ts=280.0, rh=0.0)
    c.qv[:] = F32(0.0)
    c.add("qi", 3e-4, 5000, 2000)
    c.add("qs", 3e-4, 5000, 2000)
    c.add("qg", 3e-4, 5000, 2000)
    c.add("qr", 3e-4, 1000, 1000)
    cols.append(c)
    c = base("edge-graupel-only-cold", ts=265.0, lapse=6e-3, rh=0.97)
    c.add("qg", 4e-3, 3000, 2000)
    cols.append(c)
    return cols


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261008)
    cols = make_columns(rng)
    ncol = len(cols)
    flds = [c.fields() for c in cols]
    parts = []
    for name in IN3D:     # (ncol, nk) Fortran order == (nk, ncol) C order
        parts.append(np.stack([f[name] for f in flds], axis=1).ravel())
    parts.append(np.array([c.ht for c in cols], dtype=F32))
    parts.append(np.array([c.rainnc for c in cols], dtype=F32))
    parts.append(np.array([c.snownc for c in cols], dtype=F32))
    parts.append(np.array([c.graupelnc for c in cols], dtype=F32))
    blob = np.concatenate(parts).astype("<f4")
    sets = {"main": (12, 12.0), "longdt": (6, 60.0)}
    cases = []
    for sname, (nsteps, dt) in sets.items():
        blob.tofile(out / f"{sname}.bin")
        for ihail, ice2 in COMBOS:
            cases.append({"name": f"{sname}-hail{ihail}-ice{ice2}",
                          "input": f"{sname}.bin", "ncol": ncol, "nk": NK,
                          "nsteps": nsteps, "dt": dt, "ihail": ihail,
                          "ice2": ice2})
    index = {"columns": [c.label for c in cols], "cases": cases,
             "in3d": IN3D, "in2d": IN2D}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{ncol} columns x {NK} levels, {len(cases)} cases -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
