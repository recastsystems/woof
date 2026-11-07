#!/usr/bin/env python3
"""Column fixtures for the P3 (mp_physics=50 and 53) oracle against WRF 4.6.1.

Writes the byte format of evidence/p3-fortran-oracle-20260828/FORMAT.md
(raw little-endian binary32, Fortran element order, fifteen (ncol, nk)
inputs then five (ncol,) inputs) plus one .ctl per fixture and index.json,
so the unmodified p3_ref_driver.f90 and the WOOF runner read the SAME BYTES.

Every column is built from an analytic sounding: a stretched 50-level grid,
a temperature profile with optional surface inversion and tropopause,
pressure integrated hydrostatically from a chosen surface pressure, a
relative-humidity profile, and hydrometeor layers whose amplitudes are drawn
from a seeded generator.  Nothing here calls P3 or WOOF code, so a fixture
cannot be shaped by the implementation it tests.

Fixtures written:

  R01-regimes-1step   all regime columns, dt 20 s, 1 step
  R30-regimes-30step  the same columns, dt 20 s, 30 steps (free running)
  R60-regimes-dt60    the same columns, dt 60 s, 10 steps
  E05-edges-5step     edge-case columns, dt 20 s, 5 steps
  S30-subset-30step   ten R30 columns, dt 20 s, 30 steps (committed with
                      its WRF dump as the device test's oracle)
  X01-coldstart       th_old = qv_old = 0 (WRF's first-call state); WRF
                      4.6.1 evaluates 0/0 there, WOOF floors qvs/qvi.  This
                      fixture DOCUMENTS that divergence; it is not part of
                      the 0 ULP claim.
  X02-overflow-column the one real-forecast column (tests/data/
                      p3_cold_cloud_overflow_column.npz, the cell that took
                      a 6 h forecast non-finite): WRF's immersion-freezing
                      product cons6*gamma*tmp1*dum**2 overflows float32
                      there and the conservation limiter turns the +Inf
                      into NaN; WOOF's rescue recomputes only that product.
                      Also outside the claim; it DOCUMENTS the second
                      divergence against the real Fortran.

``--variant 53`` writes the same soundings for the mp_physics=53 call shape
(prognostic droplet number, three-moment ice): every fixture gains ``nc``
and ``qzi`` (the ADVECTED ice moment, (N*Z)**0.5) after the fifteen mp=50
inputs and the dumps gain the same two fields after the seventeen mp=50
outputs (tools/p3_wrf461_oracle/p3_ref_driver_3mom.f90).  The two fields
are drawn from their OWN generator, so the mp=50 bytes of every fixture are
unchanged, and the edge set gains the three-moment edges (qzi with no ice,
ice with no qzi, mu_i pinned at both bounds, droplets with zero number,
activation in clear supersaturated air).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

RD, CP, G, P0 = 287.15, 1005.0, 9.816, 1.0e5
RCP = RD / CP
NK = 50

IN3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
        "th_old", "qv_old", "pii", "p", "dz", "w"]
IN2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]
OUT3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
         "th_old", "qv_old", "zdbz", "effc", "effi", "vmi", "di", "rhopo"]
OUT2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]
#: Fields a variant adds after the fifteen inputs and the seventeen outputs.
VARIANT_EXTRA = {50: (), 53: ("nc", "qzi")}


# Independent saturation formulas (Bolton; Murphy-Koop), used only to target
# a humidity.  Deliberately not P3's polysvp1.
def es_liq(T):
    Tc = T - 273.15
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def es_ice(T):
    return np.exp(9.550426 - 5723.265 / T + 3.53068 * np.log(T)
                  - 0.00728332 * T)


def mixr(e, p):
    return 0.622 * e / np.maximum(p - e, 1.0)


def grid(dz0=40.0, growth=1.08, dzmax=700.0, nk=NK):
    dz = np.minimum(dz0 * growth ** np.arange(nk), dzmax)
    zi = np.concatenate([[0.0], np.cumsum(dz)])
    return dz, 0.5 * (zi[1:] + zi[:-1])


class Column:
    """One sounding.  Fields are float64 until packed."""

    def __init__(self, label, regime, *, psfc, tsfc, lapse=6.5e-3,
                 ztrop=12000.0, tstrat_lapse=-1.0e-3, inv_depth=0.0,
                 inv_strength=0.0, dz0=40.0, growth=1.08, dzmax=700.0):
        self.label, self.regime = label, regime
        dz, z = grid(dz0, growth, dzmax)
        self.dz, self.z = dz, z
        T = tsfc - lapse * np.minimum(z, ztrop)
        T = T - tstrat_lapse * np.maximum(z - ztrop, 0.0)
        if inv_depth > 0.0:
            # surface-based inversion: T rises by inv_strength over inv_depth
            ramp = np.clip(z / inv_depth, 0.0, 1.0)
            T = T - inv_strength * (1.0 - ramp)
        self.T = T
        # hydrostatic pressure at mid levels
        p = np.empty(NK)
        lnp = np.log(psfc) - G * z[0] / (RD * T[0])
        p[0] = np.exp(lnp)
        for k in range(1, NK):
            Tm = 0.5 * (T[k] + T[k - 1])
            lnp -= G * (z[k] - z[k - 1]) / (RD * Tm)
            p[k] = np.exp(lnp)
        self.p = p
        self.qv = np.zeros(NK)
        self.h = {n: np.zeros(NK) for n in ("qc", "qr", "nr", "qi", "ni",
                                             "qir", "qib")}
        self.dth = np.zeros(NK)     # th - th_old  (the dynamics' tendency)
        self.dqv = np.zeros(NK)     # qv - qv_old
        self.w = np.zeros(NK)

    # humidity -----------------------------------------------------------
    def rh_liq(self, rh):
        self.qv = np.asarray(rh) * mixr(es_liq(self.T), self.p)

    def rh_mixed(self, rh):
        """rh w.r.t. liquid above 0 C and w.r.t. ice below."""
        e = np.where(self.T >= 273.15, es_liq(self.T), es_ice(self.T))
        self.qv = np.asarray(rh) * mixr(e, self.p)

    # hydrometeors -------------------------------------------------------
    def layer(self, zbot, ztop, taper=300.0):
        z = self.z
        up = np.clip((z - zbot) / taper + 1.0, 0.0, 1.0)
        dn = np.clip((ztop - z) / taper + 1.0, 0.0, 1.0)
        return np.minimum(up, dn) * ((z >= zbot - taper) & (z <= ztop + taper))

    def add_cloud(self, amount, zbot, ztop):
        self.h["qc"] += amount * self.layer(zbot, ztop)

    def add_rain(self, amount, zbot, ztop, dmean=1.0e-3):
        s = amount * self.layer(zbot, ztop)
        m = np.pi / 6.0 * 1000.0 * dmean ** 3
        self.h["qr"] += s
        self.h["nr"] += np.where(s > 0, s / m, 0.0)

    def add_ice(self, amount, zbot, ztop, *, mmean=1.0e-9, rimefrac=0.0,
                rhorime=400.0):
        s = amount * self.layer(zbot, ztop)
        self.h["qi"] += s
        self.h["ni"] += np.where(s > 0, s / mmean, 0.0)
        self.h["qir"] += rimefrac * s
        self.h["qib"] += rimefrac * s / rhorime

    def tendency(self, rng, dth=0.05, dqv_rel=2.0e-3):
        """Small dynamics tendency since the previous P3 call."""
        self.dth = rng.normal(0.0, dth, NK)
        self.dqv = rng.normal(0.0, dqv_rel, NK) * self.qv
        self.w = rng.normal(0.0, 0.5, NK)

    # output ---------------------------------------------------------------
    def fields(self):
        th = self.T / (self.p * 1.0e-5) ** RCP
        f = dict(self.h)
        f["th"] = th
        f["qv"] = self.qv
        f["th_old"] = th - self.dth
        f["qv_old"] = np.maximum(self.qv - self.dqv, 0.0)
        f["pii"] = (self.p / P0) ** RCP
        f["p"] = self.p
        f["dz"] = self.dz
        f["w"] = self.w
        return f


def U(rng, a, b):
    return float(rng.uniform(a, b))


# ---------------------------------------------------------------------------
# Regimes.  Each maker returns a list of Columns; counts are in the names.
# ---------------------------------------------------------------------------

def deep_convection(rng, n=12):
    out = []
    for j in range(n):
        c = Column(f"conv-{j:02d}", "convective", psfc=U(rng, 99000, 101500),
                   tsfc=U(rng, 295, 305), lapse=U(rng, 6.3e-3, 7.4e-3),
                   ztrop=U(rng, 12500, 15000))
        lcl = U(rng, 600, 1500)
        rh = np.where(c.z < lcl, U(rng, 0.65, 0.85),
                      U(rng, 0.97, 1.03))
        rh = np.where(c.z > 13000, 0.6, rh)
        c.rh_mixed(rh)
        c.add_cloud(U(rng, 0.5e-3, 3.0e-3), lcl, U(rng, 7000, 11000))
        c.add_rain(U(rng, 0.3e-3, 4.0e-3), 0.0, U(rng, 4000, 7000),
                   dmean=U(rng, 0.6e-3, 2.5e-3))
        c.add_ice(U(rng, 0.2e-3, 3.0e-3), U(rng, 4500, 6500),
                  U(rng, 11000, 14000), mmean=10 ** U(rng, -10, -7.5),
                  rimefrac=U(rng, 0.2, 0.9), rhorime=U(rng, 300, 900))
        c.tendency(rng, dth=0.2, dqv_rel=5e-3)
        out.append(c)
    return out


def stratiform_melting(rng, n=8):
    out = []
    for j in range(n):
        c = Column(f"strat-{j:02d}", "stratiform-melting",
                   psfc=U(rng, 98000, 102000), tsfc=U(rng, 279, 290),
                   lapse=U(rng, 5.5e-3, 6.5e-3))
        zfrz = (c.T[0] - 273.15) / 6.0e-3
        base = U(rng, 1200, 2500)
        rh = np.where(c.z < base, U(rng, 0.75, 0.9), U(rng, 0.96, 1.01))
        c.rh_mixed(rh)
        c.add_ice(U(rng, 0.1e-3, 1.0e-3), zfrz - 300, U(rng, 7000, 9000),
                  mmean=10 ** U(rng, -9.5, -7.0),
                  rimefrac=U(rng, 0.0, 0.35), rhorime=U(rng, 100, 500))
        c.add_rain(U(rng, 0.1e-3, 1.0e-3), 0.0, zfrz + 200,
                   dmean=U(rng, 0.5e-3, 1.5e-3))
        c.add_cloud(U(rng, 0.05e-3, 0.4e-3), base, zfrz + 1500)
        c.tendency(rng, dth=0.05)
        out.append(c)
    return out


def stable_night(rng, n=8):
    out = []
    for j in range(n):
        c = Column(f"night-{j:02d}", "stable-night",
                   psfc=U(rng, 99500, 103000), tsfc=U(rng, 268, 288),
                   lapse=U(rng, 5.0e-3, 6.5e-3), inv_depth=U(rng, 150, 500),
                   inv_strength=U(rng, 4, 14))
        fog = U(rng, 60, 250)
        rh = np.where(c.z < fog, U(rng, 1.0, 1.01), U(rng, 0.25, 0.6))
        c.rh_mixed(rh) if c.T[0] < 273.15 else c.rh_liq(rh)
        if j % 4 != 3:      # three in four carry fog; one is clear and dry
            c.add_cloud(U(rng, 0.05e-3, 0.4e-3), 0.0, fog)
            c.add_rain(U(rng, 1e-7, 1e-5), 0.0, fog,
                       dmean=U(rng, 0.1e-3, 0.3e-3))
        if c.T[0] < 273.15 and j % 2 == 0:
            c.add_ice(U(rng, 1e-6, 2e-5), 0.0, fog, mmean=1e-11)
        c.tendency(rng, dth=0.02, dqv_rel=1e-3)
        out.append(c)
    return out


def winter_snow(rng, n=10):
    out = []
    for j in range(n):
        c = Column(f"snow-{j:02d}", "snow-ice", psfc=U(rng, 97000, 103000),
                   tsfc=U(rng, 248, 272.5), lapse=U(rng, 4.0e-3, 6.5e-3),
                   ztrop=U(rng, 8000, 10000))
        top = U(rng, 3000, 7000)
        rhi = np.where(c.z < top, U(rng, 1.0, 1.15), U(rng, 0.5, 0.8))
        c.rh_mixed(rhi)
        if j % 5 != 4:      # four in five carry snow
            c.add_ice(U(rng, 0.05e-3, 0.8e-3), 0.0, top,
                      mmean=10 ** U(rng, -10, -7.5),
                      rimefrac=U(rng, 0.0, 0.5), rhorime=U(rng, 50, 600))
        if j % 3 == 0:      # supercooled liquid layer, mixed phase
            zb = U(rng, 500, 2000)
            c.add_cloud(U(rng, 0.05e-3, 0.3e-3), zb, zb + U(rng, 300, 900))
        if j % 4 == 1:      # freezing drizzle
            c.add_rain(U(rng, 1e-5, 2e-4), 0.0, U(rng, 800, 2000),
                       dmean=U(rng, 0.2e-3, 0.5e-3))
        c.tendency(rng, dth=0.05)
        out.append(c)
    return out


def marine_sc(rng, n=8):
    out = []
    for j in range(n):
        zinv = U(rng, 900, 1700)
        c = Column(f"ocean-{j:02d}", "ocean", psfc=U(rng, 100500, 102500),
                   tsfc=U(rng, 287, 300), lapse=U(rng, 6.0e-3, 9.0e-3))
        # capping inversion: +6..10 K jump at zinv, dry above
        jump = U(rng, 6, 10)
        c.T = np.where(c.z > zinv, c.T + jump, c.T)
        base = zinv - U(rng, 300, 700)
        rh = np.where(c.z < base, U(rng, 0.8, 0.95),
                      np.where(c.z < zinv, U(rng, 1.0, 1.01),
                               U(rng, 0.1, 0.35)))
        c.rh_liq(rh)
        c.add_cloud(U(rng, 0.2e-3, 0.8e-3), base, zinv, )
        c.add_rain(10 ** U(rng, -6, -4), 0.0, zinv,
                   dmean=U(rng, 0.08e-3, 0.4e-3))
        c.tendency(rng, dth=0.03)
        out.append(c)
    return out


def high_terrain(rng, n=8):
    out = []
    for j in range(n):
        c = Column(f"terrain-{j:02d}", "high-terrain",
                   psfc=U(rng, 58000, 76000), tsfc=U(rng, 255, 292),
                   lapse=U(rng, 5.5e-3, 8.0e-3), ztrop=U(rng, 9000, 11000),
                   dz0=U(rng, 25, 60))
        rh = np.where(c.z < U(rng, 1500, 4000), U(rng, 0.95, 1.05),
                      U(rng, 0.3, 0.7))
        c.rh_mixed(rh)
        kind = j % 4
        if kind in (0, 1):   # orographic cloud + snow/graupel
            c.add_cloud(U(rng, 0.1e-3, 1.2e-3), 200, U(rng, 1500, 3500))
            c.add_ice(U(rng, 0.1e-3, 1.5e-3), 0, U(rng, 2500, 5000),
                      mmean=10 ** U(rng, -9.5, -7),
                      rimefrac=U(rng, 0.1, 0.8), rhorime=U(rng, 200, 800))
        if kind in (1, 2):   # rain
            c.add_rain(U(rng, 0.05e-3, 1.5e-3), 0, U(rng, 1000, 3000),
                       dmean=U(rng, 0.4e-3, 1.5e-3))
        if kind == 3:        # dry, deep mixed layer afternoon
            c.rh_mixed(np.full(NK, U(rng, 0.15, 0.4)))
        c.tendency(rng, dth=0.1)
        out.append(c)
    return out


def anvil_homogeneous(rng, n=6):
    out = []
    for j in range(n):
        c = Column(f"anvil-{j:02d}", "anvil-homogeneous",
                   psfc=U(rng, 99000, 101000), tsfc=U(rng, 298, 304),
                   lapse=U(rng, 7.0e-3, 7.8e-3), ztrop=U(rng, 14500, 16500))
        rh = np.where((c.z > 9000) & (c.z < 14500), U(rng, 1.0, 1.25),
                      U(rng, 0.5, 0.9))
        c.rh_mixed(rh)
        c.add_ice(U(rng, 0.05e-3, 1.0e-3), 9000, 14500,
                  mmean=10 ** U(rng, -12, -9.5),
                  rimefrac=U(rng, 0.0, 0.2), rhorime=U(rng, 100, 400))
        # supercooled liquid lofted above the -40 C level: homogeneous freezing
        c.add_cloud(U(rng, 0.05e-3, 1.0e-3), U(rng, 9500, 10500),
                    U(rng, 11000, 12500))
        c.add_rain(U(rng, 0.05e-3, 0.8e-3), U(rng, 9500, 10500),
                   U(rng, 11000, 12500), dmean=U(rng, 0.3e-3, 1.0e-3))
        c.tendency(rng, dth=0.3, dqv_rel=1e-2)
        out.append(c)
    return out


def hail_extreme(rng, n=4):
    out = []
    for j in range(n):
        c = Column(f"hail-{j:02d}", "hail-extreme",
                   psfc=U(rng, 85000, 100000), tsfc=U(rng, 297, 308),
                   lapse=U(rng, 7.5e-3, 8.5e-3), ztrop=U(rng, 13000, 15500))
        rh = np.where(c.z < 1500, 0.6, 1.02)
        c.rh_mixed(rh)
        c.add_cloud(U(rng, 2e-3, 6e-3), 1500, 10000)
        c.add_rain(U(rng, 3e-3, 9e-3), 0, 6000, dmean=U(rng, 2e-3, 4e-3))
        c.add_ice(U(rng, 3e-3, 12e-3), 4000, 12000, mmean=10 ** U(rng, -7, -5),
                  rimefrac=U(rng, 0.85, 0.99), rhorime=U(rng, 800, 900))
        c.tendency(rng, dth=0.5, dqv_rel=1e-2)
        out.append(c)
    return out


def edge_cases(rng):
    """Named edge columns.  Each one aims at a branch or a limiter."""
    out = []

    def base(label, **kw):
        args = dict(psfc=100000.0, tsfc=290.0, lapse=6.5e-3)
        args.update(kw)
        c = Column(label, "edge", **args)
        c.rh_mixed(np.full(NK, 0.8))
        c.tendency(rng, dth=0.05)
        return c

    c = base("edge-all-zero"); c.rh_liq(np.full(NK, 0.5)); out.append(c)

    c = base("edge-qsmall-straddle")
    QS = 1.0e-14
    c.h["qc"][5:15] = QS * 0.9
    c.h["qr"][5:15] = QS * 1.1
    c.h["nr"][5:15] = 1.0
    c.h["qi"][25:35] = QS
    c.h["ni"][25:35] = 1.0e-3
    out.append(c)

    c = base("edge-rain-zero-number")
    c.h["qr"][2:20] = 1.0e-3            # nr left 0
    out.append(c)

    c = base("edge-ice-zero-number", tsfc=262.0)
    c.h["qi"][0:25] = 1.0e-4            # ni left 0
    c.rh_mixed(np.full(NK, 1.02))
    out.append(c)

    c = base("edge-ni-huge", tsfc=255.0)
    c.h["qi"][0:30] = 5.0e-4
    c.h["ni"][0:30] = 1.0e9
    out.append(c)

    c = base("edge-rime-inconsistent", tsfc=265.0)
    c.h["qi"][0:20] = 3.0e-4
    c.h["ni"][0:20] = 1.0e5
    c.h["qir"][0:10] = 6.0e-4           # qir > qi
    c.h["qib"][0:10] = 6.0e-4 / 500.0
    c.h["qir"][10:20] = 1.0e-4          # rime mass with zero volume
    out.append(c)

    c = base("edge-small-negatives", tsfc=280.0)
    for n in ("qc", "qr", "qi", "nr", "ni"):
        c.h[n][3:30:3] = -1.0e-10
    c.h["qc"][10:20] = 2.0e-4
    out.append(c)

    c = base("edge-threshold-temperatures", tsfc=290.0)
    for target in (273.15, 269.15, 233.15, 258.15):
        k = int(np.argmin(np.abs(c.T - target)))
        c.T[k] = target
    c.rh_mixed(np.full(NK, 1.0))
    c.add_cloud(3.0e-4, 0, 12000)
    c.add_rain(3.0e-4, 0, 12000)
    c.add_ice(3.0e-4, 0, 12000, rimefrac=0.5, rhorime=400.0)
    out.append(c)

    c = base("edge-supersat-150pct", tsfc=295.0)
    c.rh_liq(np.where(c.z < 3000, 1.5, 0.7))
    out.append(c)

    c = base("edge-thin-layers", tsfc=288.0, dz0=5.0, growth=1.12)
    c.rh_mixed(np.full(NK, 0.95))
    c.add_rain(2.0e-3, 0, 4000, dmean=2.5e-3)
    c.add_ice(1.0e-3, 2000, 6000, mmean=1e-6, rimefrac=0.9, rhorime=900)
    out.append(c)

    c = base("edge-huge-drops", tsfc=293.0)
    c.h["qr"][0:20] = 5.0e-3
    c.h["nr"][0:20] = 10.0
    c.h["qr"][20:30] = 1.0e-5
    c.h["nr"][20:30] = 1.0e9
    out.append(c)

    c = base("edge-tiny-cloud-cold", tsfc=262.0)
    c.rh_mixed(np.full(NK, 1.0))
    c.h["qc"][0:25] = 1.0e-8            # lamc at its upper clamp
    c.h["qc"][25:35] = 1.0e-12
    out.append(c)

    c = base("edge-cold-top", tsfc=270.0, lapse=9.0e-3, ztrop=9000.0,
             tstrat_lapse=0.0)
    c.T = np.maximum(c.T, 180.0)
    c.qv = np.full(NK, 2.0e-6)
    c.add_ice(1.0e-5, 6000, 15000, mmean=1e-12)
    out.append(c)

    c = base("edge-graupel-melt", tsfc=300.0)
    c.rh_mixed(np.full(NK, 0.9))
    c.add_ice(4.0e-3, 0, 8000, mmean=1e-6, rimefrac=0.95, rhorime=900)
    out.append(c)
    return out


def regime_columns():
    rng = np.random.default_rng(20261007)
    cols = []
    for maker in (deep_convection, stratiform_melting, stable_night,
                  winter_snow, marine_sc, high_terrain, anvil_homogeneous,
                  hail_extreme):
        cols += maker(rng)
    return cols


#: Indices into regime_columns() for the committed subset fixture.
SUBSET = (0, 5, 9, 28, 38, 46, 48, 51, 54, 60)


def g_of_mu(mu):
    """G(mu) = (6+mu)(5+mu)(4+mu)/((3+mu)(2+mu)(1+mu)), the gamma-PSD ratio
    M0*M6/M3**2, written from its definition (not from P3's G_of_mu)."""
    return ((6 + mu) * (5 + mu) * (4 + mu)) / ((3 + mu) * (2 + mu) * (1 + mu))


def add_three_moment_fields(cols, seed=53):
    """Give every column a droplet number and an advected ice moment.

    nc: the cloud water spread over droplets of one mean volume diameter
    drawn per column (6-30 um).  qzi: for a gamma PSD of shape mu and bulk
    density rho_b, M3 = 6 qi/(pi rho_b) and Z = G(mu) M3**2/N, so the
    advected (N*Z)**0.5 is G(mu)**0.5 * M3; mu and rho_b are drawn per
    column (0-14 and 80-800 kg m-3).  Columns that already carry ``nc`` or
    ``qzi`` (the named three-moment edges) keep theirs.
    """
    rng = np.random.default_rng(seed)
    for c in cols:
        dd = rng.uniform(6.0e-6, 30.0e-6)
        mu = rng.uniform(0.0, 14.0)
        rhob = rng.uniform(80.0, 800.0)
        if "nc" not in c.h:
            m = np.pi / 6.0 * 1000.0 * dd ** 3
            c.h["nc"] = np.where(c.h["qc"] > 0, c.h["qc"] / m, 0.0)
        if "qzi" not in c.h:
            m3 = 6.0 * np.maximum(c.h["qi"], 0.0) / (np.pi * rhob)
            c.h["qzi"] = np.sqrt(g_of_mu(mu)) * m3


def three_moment_edges(rng):
    """Edges aimed at the mp=53 branches the mp=50 set cannot reach."""
    out = []

    def base(label, **kw):
        args = dict(psfc=100000.0, tsfc=265.0, lapse=6.5e-3)
        args.update(kw)
        c = Column(label, "edge-3mom", **args)
        c.rh_mixed(np.full(NK, 0.9))
        c.tendency(rng, dth=0.05)
        return c

    c = base("edge3-qzi-without-ice")       # zitot zeroed where qitot < qsmall
    c.h["qi"][0:10] = 0.5e-14
    c.h["ni"][0:10] = 1.0e3
    c.h["qzi"] = np.zeros(NK)
    c.h["qzi"][0:20] = 1.0e-6
    out.append(c)

    c = base("edge3-ice-without-qzi")       # zsmall floor, then mu bounds
    c.add_ice(4.0e-4, 0, 6000, mmean=1e-9, rimefrac=0.3, rhorime=400.0)
    c.h["qzi"] = np.zeros(NK)
    out.append(c)

    c = base("edge3-mu-at-zero-bound")      # G >= 20: mu = 0, Z clamped
    c.add_ice(6.0e-4, 0, 7000, mmean=1e-8, rimefrac=0.1, rhorime=200.0)
    c.h["qzi"] = 1.0e3 * 6.0 * c.h["qi"] / (np.pi * 100.0)
    out.append(c)

    c = base("edge3-mu-at-max-bound")       # G < 1.23: mu = mu_i_max
    c.add_ice(6.0e-4, 0, 7000, mmean=1e-8, rimefrac=0.6, rhorime=700.0)
    c.h["qzi"] = 1.0e-3 * 6.0 * c.h["qi"] / (np.pi * 500.0)
    out.append(c)

    c = base("edge3-ni-below-qsmall")       # wrapper sets Z = 0 when Ni < qsmall
    c.add_ice(2.0e-4, 0, 5000, mmean=1e-9)
    c.h["ni"][0:15] = 0.5e-14
    out.append(c)

    c = base("edge3-cloud-zero-number", tsfc=285.0)
    c.rh_liq(np.full(NK, 1.005))
    c.add_cloud(5.0e-4, 300, 3000)
    c.h["nc"] = np.zeros(NK)
    out.append(c)

    c = base("edge3-cloud-huge-number", tsfc=280.0)
    c.rh_liq(np.full(NK, 1.002))
    c.add_cloud(3.0e-4, 0, 2500)
    c.h["nc"] = np.where(c.h["qc"] > 0, 5.0e9, 0.0)
    out.append(c)

    c = base("edge3-activation-clear-supersat", tsfc=290.0)
    c.rh_liq(np.where(c.z < 2500, 1.01, 0.6))
    c.h["nc"] = np.zeros(NK)
    out.append(c)

    c = base("edge3-freezing-all-paths", tsfc=300.0, lapse=7.8e-3,
             ztrop=15000.0)
    c.rh_mixed(np.full(NK, 1.02))
    c.add_cloud(1.0e-3, 2000, 12500)
    c.add_rain(1.0e-3, 2000, 12500, dmean=1.0e-3)
    c.add_ice(5.0e-4, 6000, 13000, mmean=1e-10, rimefrac=0.2)
    out.append(c)

    c = base("edge3-hail-dense", tsfc=302.0, lapse=8.0e-3, ztrop=14000.0)
    c.rh_mixed(np.where(c.z < 1500, 0.6, 1.02))
    c.add_cloud(3.0e-3, 1500, 9000)
    c.add_rain(5.0e-3, 0, 5000, dmean=3.0e-3)
    c.add_ice(8.0e-3, 3000, 11000, mmean=1e-5, rimefrac=0.97, rhorime=900)
    out.append(c)
    return out


def pack(cols, variant=50):
    ncol = len(cols)
    fs = [c.fields() for c in cols]
    blob = b""
    for name in IN3D + list(VARIANT_EXTRA[variant]):
        a = np.stack([f[name] for f in fs])          # (ncol, nk)
        blob += np.asarray(a, dtype=np.float64).astype("<f4").tobytes(
            order="F")
    for name in IN2D:
        blob += np.zeros(ncol, dtype="<f4").tobytes()
    return blob


OVERFLOW_COLUMN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "data",
    "p3_cold_cloud_overflow_column.npz")


def overflow_fixture(outdir, nsteps=2, variant=50):
    """X02: the real-forecast column whose immersion freezing overflows.

    Its state is the model's own float32 numbers at the failing call (it
    284, dt 60 s), written as stored rather than rebuilt.  ``pii`` is Exner
    from its pressure (P3 rebuilds Exner from pressure itself,
    module_mp_p3.F:2293, so mp=50 does not read it) and ``w`` is zero, as
    in every other fixture.  Both drivers count steps from 1 rather than
    from the stored 284, so step 1 takes P3's first-call branches (the
    initial saturation adjustment, no specified-Nc activation) on both
    sides alike; the overflowing immersion-freezing product does not read
    ``it``.
    """
    data = np.load(OVERFLOW_COLUMN)
    nk = int(data["th"].size)
    f = {name: data[src] for name, src in (
        ("th", "th"), ("qv", "qv"), ("qc", "qc"), ("qr", "qr"), ("nr", "nr"),
        ("qi", "qi"), ("ni", "ni"), ("qir", "qir"), ("qib", "qib"),
        ("th_old", "th_old"), ("qv_old", "qv_old"), ("p", "pres"),
        ("dz", "dz"))}
    f["pii"] = ((f["p"].astype(np.float64) / P0) ** RCP).astype(np.float32)
    f["w"] = np.zeros(nk, dtype=np.float32)
    if variant == 53:
        # The stored column is an mp=50 state: give it a droplet number and
        # an advected ice moment the way add_three_moment_fields does.
        qc = f["qc"].astype(np.float64)
        f["nc"] = np.where(qc > 0, qc / (np.pi / 6.0 * 1000.0 * 12e-6 ** 3),
                           0.0)
        f["qzi"] = (np.sqrt(g_of_mu(4.0)) * 6.0
                    * np.maximum(f["qi"].astype(np.float64), 0.0)
                    / (np.pi * 300.0))
    blob = b""
    for name in IN3D + list(VARIANT_EXTRA[variant]):
        blob += np.asarray(f[name], dtype="<f4").tobytes()
    for name in IN2D:
        blob += np.zeros(1, dtype="<f4").tobytes()
    name = "X02-overflow-column"
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


def main(outdir, variant=50):
    os.makedirs(outdir, exist_ok=True)
    regimes = regime_columns()
    edges = edge_cases(np.random.default_rng(7))
    if variant == 53:
        edges = edges + three_moment_edges(np.random.default_rng(530))
        add_three_moment_fields(regimes)
        add_three_moment_fields(edges, seed=531)
    fixtures = [
        ("R01-regimes-1step", regimes, 20.0, 1, False),
        ("R30-regimes-30step", regimes, 20.0, 30, False),
        ("R60-regimes-dt60", regimes, 60.0, 10, False),
        ("E05-edges-5step", edges, 20.0, 5, False),
        # Ten columns of R30, small enough to commit with their WRF dump
        # (tests/data/oracles/p3/): every regime with rain or ice, plus
        # snow-00, the column the default build's subnormal flush moved.
        ("S30-subset-30step", [regimes[i] for i in SUBSET], 20.0, 30, False),
        ("X01-coldstart", regimes[:4] + edges[:2], 20.0, 2, True),
    ]
    index = []
    for name, cols, dt, nsteps, cold in fixtures:
        blob = pack(cols, variant)
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
        with open(os.path.join(outdir, name + ".ctl"), "w", newline="\n") as fh:
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
    index.append(overflow_fixture(outdir, variant=variant))
    extra = list(VARIANT_EXTRA[variant])
    with open(os.path.join(outdir, "index.json"), "w", newline="\n") as fh:
        json.dump(dict(variant=variant, in3d=IN3D + extra, in2d=IN2D,
                       out3d=OUT3D + extra, out2d=OUT2D, cases=index),
                  fh, indent=1)
    for row in index:
        print("%-20s ncol=%3d nk=%d nsteps=%2d dt=%4.0f %s"
              % (row["name"], row["ncol"], row["nk"], row["nsteps"],
                 row["dt"], row["sha256"][:16]))
    regs = {}
    for c in regimes + edges:
        regs[c.regime] = regs.get(c.regime, 0) + 1
    print("columns by regime:", regs, "total", sum(regs.values()))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", default="fixtures")
    ap.add_argument("--variant", type=int, default=50,
                    choices=sorted(VARIANT_EXTRA))
    a = ap.parse_args()
    main(a.outdir, a.variant)
