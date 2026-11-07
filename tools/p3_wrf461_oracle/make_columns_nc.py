#!/usr/bin/env python3
"""Column fixtures for the P3 mp_physics = 51 (prognostic Nc) oracle.

    python make_columns_nc.py <outdir>

The mp = 50 regime and edge columns of ``make_columns.py`` (the same seeded
soundings, byte for byte) each gain a droplet-number profile, and a set of
droplet-number edge columns is added.  The byte format is the mp = 50
FORMAT.md layout with ONE extra (ncol, nk) array, ``nc`` [# kg-1], after
the fifteen three-dimensional inputs; dumps carry ``nc`` after the
seventeen three-dimensional outputs (``p3_nc_ref_driver.f90``,
``run_woof.py --predict-nc``).  ``index.json`` names both lists.

Nothing here calls P3 or WOOF code.  Droplet number comes from a chosen
mean droplet radius per regime (maritime large and few, continental and
polluted small and many), so a fixture cannot be shaped by the
implementation it tests.

Fixtures:

  N01-regimes-1step   64 regime columns, dt 20 s, 1 step (it = 1: the
                      activation number is computed, its mass is not)
  N30-regimes-30step  the same, 30 steps (it > 1 from step 2)
  N60-regimes-dt60    the same, dt 60 s, 10 steps
  NE5-edges-5step     the 14 mp = 50 edge columns plus 14 droplet-number
                      edge columns, dt 20 s, 5 steps
  NS30-subset-30step  ten N30 columns (committed with their WRF dump)
  NA03-activation     activation sweeps: supersaturation from 2e-6 to 0.6
                      across levels, warm and cold, clean and polluted,
                      dt 20 s, 3 steps (every DERF branch is reached)
  NX01-coldstart      th_old = qv_old = 0 (outside the claim, the mp = 50
                      X01 divergence)
  NX02-overflow-column the real-forecast immersion-freezing overflow column
                      with a droplet profile (outside the claim, the mp = 50
                      X02 divergence)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_columns as MC  # noqa: E402

NK = MC.NK
IN3D = MC.IN3D + ["nc"]
IN2D = MC.IN2D
OUT3D = MC.OUT3D + ["nc"]
OUT2D = MC.OUT2D
RHOW = 1000.0

#: Mean droplet radius [m] per regime: the number profile is qc over the
#: mass of that droplet.  Ranges, not single values, drawn per column.
REGIME_RADIUS = {
    "convective": (6.0e-6, 11.0e-6),
    "stratiform-melting": (8.0e-6, 14.0e-6),
    "stable-night": (4.0e-6, 9.0e-6),
    "snow-ice": (7.0e-6, 15.0e-6),
    "ocean": (10.0e-6, 18.0e-6),
    "high-terrain": (6.0e-6, 12.0e-6),
    "anvil-homogeneous": (4.0e-6, 8.0e-6),
    "hail-extreme": (6.0e-6, 10.0e-6),
    "edge": (8.0e-6, 12.0e-6),
}


def droplet_number(qc, radius):
    """qc / (4/3 pi rhow r^3), zero where qc is not positive."""
    m = 4.0 / 3.0 * np.pi * RHOW * radius ** 3
    return np.where(qc > 0.0, qc / m, 0.0)


def with_nc(cols, rng, *, residual_every=5):
    """Give every column a droplet number.  Every ``residual_every``-th
    column also keeps a small droplet number where it has no cloud (a
    residual the scheme must clip, module_mp_p3.F:2365-2370)."""
    for j, c in enumerate(cols):
        lo, hi = REGIME_RADIUS[c.regime]
        r = MC.U(rng, lo, hi)
        nc = droplet_number(c.h["qc"], r)
        if residual_every and j % residual_every == residual_every - 1:
            nc = np.where(nc > 0.0, nc, MC.U(rng, 1.0e3, 1.0e6))
        c.h["nc"] = nc
    return cols


#: Flatau, Walko and Cotton (1992) eighth-order fit for saturation vapour
#: pressure over liquid [hPa], the published coefficients, in float64.
#: Used only to PLACE a supersaturation: the scheme diagnoses its own from
#: its own float32 evaluation, so a target lands within rounding of where
#: it is aimed rather than ~1e-3 away, as the Bolton formula would put it.
FLATAU_LIQ = (6.11239921, 0.443987641, 0.142986287e-1, 0.264847430e-3,
              0.302950461e-5, 0.206739458e-7, 0.640689451e-10,
              -0.952447341e-13, -0.976195544e-15)


def es_flatau_liq(T):
    dt = np.asarray(T, dtype=np.float64) - 273.15
    acc = np.zeros_like(dt)
    for a in reversed(FLATAU_LIQ):
        acc = acc * dt + a
    return acc * 100.0


def sat_qv(c, T=None):
    """Saturation mixing ratio over liquid (float64 Flatau fit)."""
    T = c.T if T is None else T
    return MC.mixr(es_flatau_liq(T), c.p)


def nc_edge_cases(rng):
    """Named droplet-number edge columns.  Each aims at a branch."""
    out = []

    def base(label, **kw):
        args = dict(psfc=100000.0, tsfc=290.0, lapse=6.5e-3)
        args.update(kw)
        c = MC.Column(label, "edge", **args)
        c.rh_mixed(np.full(NK, 0.8))
        c.tendency(rng, dth=0.05)
        c.h["nc"] = np.zeros(NK)
        return c

    # cloud with no droplets: get_cloud_dsd2's nsmall floor, then the lambda
    # limiter sets nc (:6655, :6679-6685)
    c = base("nc-zero-with-cloud"); c.rh_liq(np.full(NK, 1.0))
    c.add_cloud(4.0e-4, 300, 3000); out.append(c)

    c = base("nc-one-per-kg"); c.rh_liq(np.full(NK, 1.0))
    c.add_cloud(3.0e-4, 300, 3000)
    c.h["nc"] = np.where(c.h["qc"] > 0, 1.0, 0.0); out.append(c)

    c = base("nc-polluted-5e9"); c.rh_liq(np.full(NK, 1.0))
    c.add_cloud(2.0e-4, 300, 3000)
    c.h["nc"] = np.where(c.h["qc"] > 0, 5.0e9, 0.0); out.append(c)

    c = base("nc-without-cloud")
    c.h["nc"][:] = 1.0e8; out.append(c)

    # droplets already above the aerosol total (nanew1 = 300e6 /kg): the
    # activation tendency is limited to zero (:3327-3329)
    c = base("nc-above-aerosol-total")
    c.rh_liq(np.where(c.z < 4000, 1.02, 0.7))
    c.add_cloud(5.0e-4, 200, 3500)
    c.h["nc"] = np.where(c.h["qc"] > 0, 4.0e8, 0.0); out.append(c)

    # supersaturation straddling the 1e-6 activation threshold (:3316):
    # alternate levels aimed at 0.5e-6 and 2e-6
    c = base("nc-sup-threshold-straddle")
    qs = sat_qv(c)
    sup = np.where(np.arange(NK) % 2 == 0, 0.5e-6, 2.0e-6)
    c.qv = qs * (1.0 + sup); c.dth[:] = 0.0; c.dqv[:] = 0.0
    c.add_cloud(1.0e-4, 0, 2000)
    c.h["nc"] = droplet_number(c.h["qc"], 10e-6); out.append(c)

    # drizzle: few large droplets, strong autoconversion and accretion
    c = base("nc-drizzle", tsfc=288.0)
    c.rh_liq(np.where(c.z < 1500, 1.005, 0.4))
    c.add_cloud(1.0e-3, 200, 1400)
    c.add_rain(2.0e-5, 0, 1400, dmean=0.15e-3)
    c.h["nc"] = droplet_number(c.h["qc"], 22e-6); out.append(c)

    # polluted deep cloud, mixed phase: collection by ice, immersion freezing
    c = base("nc-polluted-mixed", tsfc=296.0)
    c.rh_mixed(np.where(c.z < 9000, 1.01, 0.6))
    c.add_cloud(2.0e-3, 800, 9000)
    c.add_ice(1.0e-3, 4500, 11000, mmean=1e-9, rimefrac=0.5, rhorime=500)
    c.add_rain(1.0e-3, 0, 5000, dmean=1.2e-3)
    c.h["nc"] = droplet_number(c.h["qc"], 5e-6); out.append(c)

    # homogeneous freezing of droplets carrying number (:4540-4575)
    c = base("nc-homogeneous-freeze", tsfc=300.0, lapse=7.6e-3,
             ztrop=15500.0)
    c.rh_mixed(np.where((c.z > 9000) & (c.z < 14000), 1.1, 0.6))
    c.add_cloud(5.0e-4, 9800, 12500)
    c.h["nc"] = droplet_number(c.h["qc"], 6e-6); out.append(c)

    # cloud above 0 C touching melting ice: shedding source ncshdc
    c = base("nc-shedding", tsfc=301.0)
    c.rh_mixed(np.full(NK, 0.99))
    c.add_cloud(1.0e-3, 0, 3000)
    c.add_ice(2.0e-3, 0, 6000, mmean=1e-7, rimefrac=0.9, rhorime=850)
    c.h["nc"] = droplet_number(c.h["qc"], 9e-6); out.append(c)

    # thin layers: two-moment cloud sedimentation substeps (:4009-4066)
    c = base("nc-thin-layers", tsfc=286.0, dz0=4.0, growth=1.12)
    c.rh_liq(np.full(NK, 1.0))
    c.add_cloud(2.0e-3, 0, 1500)
    c.h["nc"] = droplet_number(c.h["qc"], 25e-6); out.append(c)

    # small negative droplet numbers beside cloud
    c = base("nc-small-negatives", tsfc=282.0)
    c.add_cloud(2.0e-4, 0, 2500)
    c.h["nc"] = droplet_number(c.h["qc"], 10e-6)
    c.h["nc"][2:30:3] = -1.0e-3; out.append(c)

    # cloud water at the droplet-size clamp with prognostic number
    c = base("nc-tiny-cloud", tsfc=275.0)
    c.rh_mixed(np.full(NK, 1.0))
    c.h["qc"][0:20] = 1.0e-9
    c.h["nc"][0:20] = 5.0e7
    c.h["qc"][20:30] = 1.0e-13
    c.h["nc"][20:30] = 1.0e3; out.append(c)

    # fog under an inversion, polluted
    c = base("nc-polluted-fog", tsfc=276.0, inv_depth=300.0,
             inv_strength=8.0)
    c.rh_liq(np.where(c.z < 150, 1.003, 0.5))
    c.add_cloud(3.0e-4, 0, 150)
    c.h["nc"] = droplet_number(c.h["qc"], 3.5e-6); out.append(c)
    return out


def activation_sweeps(rng):
    """Supersaturation swept log-uniformly over the column, so the
    activation's erf argument walks through every DERF branch (both
    polynomial tables, both signs).  Warm and cold, clean and polluted."""
    out = []
    specs = [("act-warm-clean", 293.0, 0.0),
             ("act-warm-polluted", 293.0, 2.0e8),
             ("act-cool-clean", 281.0, 0.0),
             ("act-cold-mixed", 268.0, 5.0e7),
             ("act-warm-cloudy", 299.0, None),
             ("act-high-terrain", 284.0, 1.0e7)]
    for label, tsfc, nc0 in specs:
        psfc = 70000.0 if "terrain" in label else 100500.0
        c = MC.Column(label, "edge", psfc=psfc, tsfc=tsfc, lapse=5.0e-3)
        sup = np.exp(np.linspace(np.log(2.0e-6), np.log(0.6), NK))
        if "cold" in label:
            sup = sup[::-1]
        c.qv = sat_qv(c) * (1.0 + sup)
        c.tendency(rng, dth=0.01, dqv_rel=1e-5)
        c.dth[:] = 0.0         # th_old = th and qv_old = qv: the scheme's
        c.dqv[:] = 0.0         # sup is the swept value
        if nc0 is None:
            c.add_cloud(5.0e-4, 0, 8000)
            c.h["nc"] = droplet_number(c.h["qc"], 9e-6)
        else:
            c.h["nc"] = np.full(NK, nc0)
            c.add_cloud(1.0e-6, 0, 4000)
        out.append(c)
    return out


def pack(cols):
    ncol = len(cols)
    fs = []
    for c in cols:
        f = c.fields()
        f["nc"] = c.h["nc"]
        fs.append(f)
    blob = b""
    for name in IN3D:
        a = np.stack([f[name] for f in fs])
        blob += np.asarray(a, dtype=np.float64).astype("<f4").tobytes(
            order="F")
    for name in IN2D:
        blob += np.zeros(ncol, dtype="<f4").tobytes()
    return blob


def overflow_fixture(outdir, nsteps=2):
    """NX02: the mp = 50 X02 column with a 10 micron droplet profile."""
    data = np.load(MC.OVERFLOW_COLUMN)
    nk = int(data["th"].size)
    f = {name: data[src] for name, src in (
        ("th", "th"), ("qv", "qv"), ("qc", "qc"), ("qr", "qr"), ("nr", "nr"),
        ("qi", "qi"), ("ni", "ni"), ("qir", "qir"), ("qib", "qib"),
        ("th_old", "th_old"), ("qv_old", "qv_old"), ("p", "pres"),
        ("dz", "dz"))}
    f["pii"] = ((f["p"].astype(np.float64) / MC.P0) ** MC.RCP).astype(
        np.float32)
    f["w"] = np.zeros(nk, dtype=np.float32)
    f["nc"] = droplet_number(f["qc"].astype(np.float64), 10e-6).astype(
        np.float32)
    blob = b""
    for name in IN3D:
        blob += np.asarray(f[name], dtype="<f4").tobytes()
    for name in IN2D:
        blob += np.zeros(1, dtype="<f4").tobytes()
    name = "NX02-overflow-column"
    dt = float(data["dt"])
    with open(os.path.join(outdir, name + ".bin"), "wb") as fh:
        fh.write(blob)
    with open(os.path.join(outdir, name + ".ctl"), "w", newline="\n") as fh:
        fh.write("@TABLEDIR@\n@FIXDIR@/%s.bin\n@OUTDIR@/%s.ref\n"
                 % (name, name))
        fh.write("%d %d %d\n%.9g\n" % (1, nk, nsteps, dt))
    return dict(name=name, ncol=1, nk=nk, dt=dt, nsteps=nsteps,
                bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                in_claim=False,
                columns=[dict(label="real-forecast-overflow",
                              regime="outside-claim",
                              psfc=round(float(f["p"][0]), 1),
                              tsfc=None, tmin=None)])


#: Indices into the regime columns for the committed subset: one of every
#: regime, the residual-droplet columns among them.
SUBSET = (0, 4, 9, 14, 23, 29, 38, 46, 56, 61)


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20261008)
    regimes = with_nc(MC.regime_columns(), rng)
    edges = with_nc(MC.edge_cases(np.random.default_rng(7)), rng,
                    residual_every=3) + nc_edge_cases(rng)
    sweeps = activation_sweeps(rng)
    fixtures = [
        ("N01-regimes-1step", regimes, 20.0, 1, False),
        ("N30-regimes-30step", regimes, 20.0, 30, False),
        ("N60-regimes-dt60", regimes, 60.0, 10, False),
        ("NE5-edges-5step", edges, 20.0, 5, False),
        ("NS30-subset-30step", [regimes[i] for i in SUBSET], 20.0, 30, False),
        ("NA03-activation", sweeps, 20.0, 3, False),
        ("NX01-coldstart", regimes[:4] + edges[-4:], 20.0, 2, True),
    ]
    index = []
    for name, cols, dt, nsteps, cold in fixtures:
        blob = pack(cols)
        if cold:
            ncol = len(cols)
            buf = np.frombuffer(blob, dtype="<f4").copy()
            per = ncol * NK
            for fname in ("th_old", "qv_old"):
                i = IN3D.index(fname)
                buf[i * per:(i + 1) * per] = 0.0
            blob = buf.tobytes()
        with open(os.path.join(outdir, name + ".bin"), "wb") as fh:
            fh.write(blob)
        with open(os.path.join(outdir, name + ".ctl"), "w",
                  newline="\n") as fh:
            fh.write("@TABLEDIR@\n@FIXDIR@/%s.bin\n@OUTDIR@/%s.ref\n"
                     % (name, name))
            fh.write("%d %d %d\n%.9g\n" % (len(cols), NK, nsteps, dt))
        index.append(dict(
            name=name, ncol=len(cols), nk=NK, dt=dt, nsteps=nsteps,
            bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
            in_claim=not cold,
            columns=[dict(label=c.label, regime=c.regime,
                          psfc=round(float(c.p[0]), 1),
                          tsfc=round(float(c.T[0]), 2),
                          tmin=round(float(c.T.min()), 2))
                     for c in cols]))
    index.append(overflow_fixture(outdir))
    with open(os.path.join(outdir, "index.json"), "w", newline="\n") as fh:
        json.dump(dict(predict_nc=True, in3d=IN3D, in2d=IN2D, out3d=OUT3D,
                       out2d=OUT2D, cases=index), fh, indent=1)
    for row in index:
        print("%-22s ncol=%3d nk=%d nsteps=%2d dt=%4.0f %s"
              % (row["name"], row["ncol"], row["nk"], row["nsteps"],
                 row["dt"], row["sha256"][:16]))
    regs = {}
    for c in regimes + edges + sweeps:
        regs[c.regime] = regs.get(c.regime, 0) + 1
    print("columns by regime:", regs, "total", sum(regs.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures-nc")
