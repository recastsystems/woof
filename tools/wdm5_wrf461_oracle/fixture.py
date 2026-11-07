"""Column fixture for the WDM5 WRF v4.6.1 oracle.

usage: python3 fixture.py OUTDIR

Writes one input file per configuration (``CONFIGS``) and ``columns.txt``
naming every column.  The same columns of 50 levels go into every
configuration; the configurations vary the time step, the number of
consecutive calls and the cold-start CCN fill.  One more configuration
(``DEEP_CONFIG``) builds the same columns on 72 levels, which WOOF compiles
as its second kernel tier (WDM5_KMAX = 80).

Columns are built hydrostatically on a stretched grid (50 m at the surface
to about 1.1 km at 17 km) from a surface pressure, a temperature profile and
a relative-humidity profile, then carry hydrometeor layers chosen per
regime.  Regimes: deep convective under polluted and clean CCN, warm
shallow cloud with no rain (the autoconversion start), stable night, winter
snow and ice, stratiform ice with a melting layer, high terrain, deep cold
anvil (homogeneous freezing), rain evaporating into dry air, numberless rain
and cloud, random mixtures, and edge cases (zeros, negatives from advection,
CCN at zero and far above any background, subnormals, exact thresholds,
exactly 273.15 K, enormous and tiny drops, strong supersaturation).  WDM5
has five mass classes (no graupel) and no land/sea dependence.  Seeded, so
the files are reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle_io import Case, write_input  # noqa: E402

NZ = 50
DEEP_NZ = 72
RD, RV, CP, G = 287.0, 461.6, 1004.5, 9.81
EP2 = RD / RV
CCN0 = 1.0e8

# name: delt, nsteps, itimestep0
CONFIGS = {
    "dt20": (20.0, 1, 2),
    "dt60_x10": (60.0, 10, 2),
    "dt300": (300.0, 1, 2),
    "dt150_x3": (150.0, 3, 2),
    "dt20_coldstart": (20.0, 1, 1),
    "dt10_x30": (10.0, 30, 2),
}
#: name: (delt, nsteps, itimestep0) on DEEP_NZ levels.
DEEP_CONFIG = ("deep72_dt30_x2", (30.0, 2, 2))


def _grid(nz=NZ):
    growth = 1.065 if nz == NZ else 1.04
    dz = 50.0 * growth ** np.arange(nz)
    zi = np.concatenate([[0.0], np.cumsum(dz)])
    zm = 0.5 * (zi[1:] + zi[:-1])
    return dz, zm


DZ, ZM = _grid()
NZ_ACTIVE = NZ


def use_grid(nz):
    """Rebuild the module grid every column constructor reads."""
    global NZ_ACTIVE, DZ, ZM
    NZ_ACTIVE = nz
    DZ, ZM = _grid(nz)


def _esat_w(t):
    return 610.78 * np.exp(17.27 * (t - 273.15) / (t - 35.86))


def _esat_i(t):
    return 610.78 * np.exp(21.875 * (t - 273.15) / (t - 7.66))


def _profile(ps, tsfc, lapse=6.5e-3, ztrop=12000.0, tstrat=None,
             inversion=None):
    """Temperature (K) and pressure (Pa) at the mass levels."""
    t = tsfc - lapse * ZM
    if tstrat is None:
        tstrat = tsfc - lapse * ztrop
    t = np.where(ZM > ztrop, tstrat, t)
    if inversion is not None:
        depth, strength = inversion
        t = t + np.where(ZM < depth, strength * (1.0 - ZM / depth), 0.0)
    p = np.empty(NZ_ACTIVE)
    pk = ps
    zprev, tprev = 0.0, t[0]
    for k in range(NZ_ACTIVE):
        tbar = 0.5 * (tprev + t[k])
        pk = pk * np.exp(-G * (ZM[k] - zprev) / (RD * tbar))
        p[k] = pk
        zprev, tprev = ZM[k], t[k]
    return t, p


def _blob(center, width, peak):
    return peak * np.exp(-0.5 * ((ZM - center) / width) ** 2)


class Column:
    def __init__(self, name, ps, tsfc, rh, **kw):
        t, p = _profile(ps, tsfc, **kw)
        es = _esat_w(t)
        qsat = EP2 * es / np.maximum(p - es, 1.0)
        rh = np.broadcast_to(rh, (NZ_ACTIVE,)) if np.ndim(rh) == 0 else rh
        self.name = name
        self.t = t
        self.p = p
        self.qv = rh * qsat
        z = np.zeros(NZ_ACTIVE)
        self.qc, self.qr, self.qi, self.qs = (z.copy() for _ in range(4))
        self.nn = np.full(NZ_ACTIVE, CCN0)
        self.nc, self.nr = z.copy(), z.copy()
        self.overrides = {}

    def saturate(self, mask, factor=1.002):
        es = _esat_w(self.t)
        self.qv[mask] = (factor * EP2 * es / (self.p - es))[mask]

    def finish(self):
        tv = self.t * (1.0 + 0.608 * self.qv)
        den = self.p / (RD * tv)
        pii = (self.p / 1.0e5) ** (RD / CP)
        th = self.t / pii
        out = {"th": th, "qv": self.qv, "qc": self.qc, "qr": self.qr,
               "qi": self.qi, "qs": self.qs, "nn": self.nn,
               "nc": self.nc, "nr": self.nr, "den": den, "pii": pii,
               "p": self.p, "delz": DZ.copy()}
        out = {k: np.asarray(v, np.float64).copy() for k, v in out.items()}
        for (field, k), value in self.overrides.items():
            out[field][k] = value
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


def rh_profile(surface, mid, top, zmid=4000.0):
    return np.interp(ZM, [0.0, zmid, 16000.0, 20000.0],
                     [surface, mid, top, top])


def convective(name, rng, scale, ccn, ps=100500.0, tsfc=303.0):
    c = Column(name, ps, tsfc, rh_profile(0.85, 0.95, 0.4))
    c.nn[:] = ccn
    c.qc[:] = _blob(3500.0, 1800.0, 1.8e-3 * scale)
    c.nc[:] = np.where(c.qc > 1e-7, ccn * (0.3 + 0.4 * rng.random()), 0.0)
    c.qr[:] = _blob(2000.0, 1500.0, 2.0e-3 * scale)
    c.nr[:] = np.where(c.qr > 1e-9, 10 ** rng.uniform(2.5, 4.5), 0.0)
    c.qs[:] = _blob(8000.0, 2200.0, 1.6e-3 * scale)
    c.qi[:] = _blob(11500.0, 1500.0, 1.5e-4 * scale)
    c.saturate((ZM > 1000.0) & (ZM < 9000.0))
    return c


def build_columns(seed=20261008):
    rng = np.random.default_rng(seed)
    cols = []
    # A. deep convective, polluted CCN
    for i, scale in enumerate((0.2, 0.5, 1.0, 1.5, 2.5, 4.0, 0.05, 1.0)):
        cols.append(convective(f"conv_polluted_{i}", rng, scale,
                               ccn=10 ** rng.uniform(8.8, 9.6)))
    # B. deep convective, clean CCN
    for i, scale in enumerate((0.3, 0.8, 1.2, 2.0, 3.0, 0.1)):
        cols.append(convective(f"conv_clean_{i}", rng, scale,
                               ccn=10 ** rng.uniform(7.3, 7.9), tsfc=301.0))
    # C. warm shallow cloud with no rain: the autoconversion start
    for i in range(8):
        c = Column(f"warm_shallow_{i}", 101000.0, 297.0 + i,
                   rh_profile(0.9, 0.7, 0.3, 2500.0))
        qc_peak = (0.4 + 0.45 * i) * 1e-3
        c.qc[:] = _blob(1500.0, 500.0, qc_peak)
        nc = 10 ** (7.3 + 0.25 * i)
        c.nc[:] = np.where(c.qc > 1e-8, nc, 0.0)
        c.nn[:] = 10 ** (7.5 + 0.3 * i)
        c.saturate(c.qc > 1e-6, 1.0 + 0.001 * (i % 4))
        cols.append(c)
    # D. stable night: inversion, fog or clear
    for i in range(6):
        tsfc = 268.0 + 4.0 * i
        c = Column(f"stable_night_{i}", 101500.0, tsfc,
                   rh_profile(0.99 if i % 2 == 0 else 0.7, 0.5, 0.2, 1500.0),
                   inversion=(400.0, 6.0))
        if i % 2 == 0:
            n = 4
            es = _esat_w(c.t[:n])
            c.qv[:n] = (1.0 + 0.002 * (i + 1)) * EP2 * es / (c.p[:n] - es)
            c.qc[:n] = 1.0e-4 * (1 + i)
            c.nc[:n] = 1.5e8
        cols.append(c)
    # E. winter snow and ice
    for i in range(8):
        tsfc = 255.0 + 2.5 * i
        c = Column(f"winter_{i}", 100000.0, tsfc,
                   rh_profile(0.95, 0.9, 0.3), lapse=5.0e-3)
        c.qs[:] = _blob(2500.0, 2000.0, (0.2 + 0.15 * i) * 1e-3)
        c.qi[:] = _blob(5000.0, 2000.0, (0.02 + 0.03 * i) * 1e-3)
        if i % 3 == 0:     # supercooled liquid layer
            c.qc[:] = _blob(1500.0, 400.0, 2.0e-4)
            c.nc[:] = np.where(c.qc > 1e-8, 8.0e7, 0.0)
        if i % 4 == 1:     # supercooled drizzle
            c.qr[:] = _blob(1000.0, 600.0, 1.0e-4)
            c.nr[:] = np.where(c.qr > 1e-9, 3.0e4, 0.0)
        es_i = _esat_i(c.t)
        ice = (ZM > 3000.0) & (ZM < 7000.0)
        c.qv[ice] = (1.05 * EP2 * es_i / (c.p - es_i))[ice]
        cols.append(c)
    # F. stratiform ice over a melting layer
    for i in range(8):
        tsfc = 276.0 + 1.5 * i
        c = Column(f"stratiform_melt_{i}", 99000.0, tsfc,
                   rh_profile(0.9, 0.98, 0.5))
        frz = (tsfc - 273.15) / 6.5e-3
        c.qs[:] = np.where(ZM > frz - 300.0, _blob(frz + 1500.0, 1500.0,
                                                   (0.3 + 0.25 * i) * 1e-3), 0.0)
        c.qi[:] = _blob(frz + 4000.0, 1500.0, (0.05 + 0.03 * i) * 1e-3)
        c.qr[:] = np.where(ZM < frz + 200.0, (0.2 + 0.1 * i) * 1e-3, 0.0)
        c.nr[:] = np.where(c.qr > 0, 10 ** (3.0 + 0.2 * i), 0.0)
        c.qc[:] = _blob(frz + 500.0, 600.0, 1.0e-4)
        c.nc[:] = np.where(c.qc > 1e-8, 2.0e8, 0.0)
        cols.append(c)
    # G. high terrain (surface 600-700 hPa)
    for i in range(6):
        ps = 60000.0 + 2000.0 * i
        if i < 3:
            c = convective(f"terrain_conv_{i}", rng, 0.5 + 0.5 * i,
                           ccn=3.0e8, ps=ps, tsfc=288.0 + 2 * i)
        else:
            c = Column(f"terrain_snow_{i}", ps, 266.0 + 2 * i,
                       rh_profile(0.97, 0.9, 0.3))
            c.qs[:] = _blob(1500.0, 1200.0, 8e-4)
            c.qi[:] = _blob(3500.0, 1500.0, 1e-4)
        cols.append(c)
    # H. deep cold anvil: liquid colder than -40 C (homogeneous freezing)
    for i in range(4):
        c = Column(f"cold_anvil_{i}", 100000.0, 300.0,
                   rh_profile(0.8, 0.9, 0.95), tstrat=200.0 - 3 * i,
                   ztrop=13000.0 + 500 * i)
        c.qc[:] = np.where(c.t < 233.0, _blob(11500.0, 1200.0, 2e-4), 0.0)
        c.nc[:] = np.where(c.qc > 1e-9, 5e7, 0.0)
        c.qi[:] = _blob(12000.0, 1500.0, 3e-4)
        c.qs[:] = _blob(10000.0, 1500.0, 4e-4)
        cols.append(c)
    # I. rain evaporating into dry air below cloud base
    for i in range(4):
        c = Column(f"evap_rain_{i}", 98000.0, 306.0 + i,
                   rh_profile(0.25 + 0.1 * i, 0.6, 0.3), lapse=8.5e-3)
        c.qr[:] = np.where(ZM < 4000.0, (0.3 + 0.4 * i) * 1e-3, 0.0)
        c.nr[:] = np.where(c.qr > 0, 10 ** (2.0 + 0.7 * i), 0.0)
        c.qc[:] = _blob(4500.0, 400.0, 3e-4)
        c.nc[:] = np.where(c.qc > 1e-8, 2e8, 0.0)
        cols.append(c)
    # J. numberless rain / numberless cloud (section 6 regime)
    for i in range(4):
        c = convective(f"numberless_{i}", rng, 0.6, ccn=2.0e8,
                       tsfc=296.0 - 10.0 * i)
        if i % 2 == 0:
            c.nr[::3] = 0.0           # rain mass without number
            c.qr[:] = np.maximum(c.qr, 1.3e-13)
        else:
            c.nc[::2] = 0.0           # cloud mass without number
        c.qv[:15] *= 0.6
        cols.append(c)
    # K. random mixtures (log-uniform amplitudes, random heights)
    for i in range(16):
        c = Column(f"random_{i}", rng.uniform(70000.0, 103000.0),
                   rng.uniform(262.0, 308.0),
                   rh_profile(rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.02),
                              rng.uniform(0.1, 0.9)))
        for field, lo, hi in (("qc", -6, -2.7), ("qr", -7, -2.3),
                              ("qi", -7, -3.5), ("qs", -7, -2.4)):
            if rng.random() < 0.8:
                arr = getattr(c, field)
                arr[:] = _blob(rng.uniform(500, 12000), rng.uniform(300, 3000),
                               10 ** rng.uniform(lo, hi))
        c.nc[:] = np.where(c.qc > 1e-10, 10 ** rng.uniform(6, 9.5), 0.0)
        c.nr[:] = np.where(c.qr > 1e-10, 10 ** rng.uniform(0, 6), 0.0)
        c.nn[:] = 10 ** rng.uniform(7.0, 10.5)
        cols.append(c)
    # L. edge cases
    cols.append(Column("edge_all_zero", 100000.0, 290.0, 0.5))
    cols.append(Column("edge_bone_dry", 100000.0, 300.0, 0.0))
    neg = convective("edge_negative_from_advection", rng, 0.5, 2e8)
    neg.qc[::4] = -1.0e-6
    neg.qr[1::4] = -3.0e-7
    neg.qi[2::4] = -1.0e-8
    neg.qs[3::4] = -2.0e-7
    neg.nn[::5] = -4.0e6
    neg.nc[::3] = -1.0e5
    neg.nr[1::3] = -10.0
    cols.append(neg)
    ccnlo = convective("edge_ccn_zero", rng, 0.7, 2e7)
    ccnlo.nn[::2] = 0.0
    cols.append(ccnlo)
    cols.append(convective("edge_ccn_extreme", rng, 0.7, 5e10))
    sub = Column("edge_subnormals", 100000.0, 285.0, 0.7)
    for field in ("qc", "qr", "qi", "qs", "nc", "nr"):
        getattr(sub, field)[:] = np.float32(1.0e-40)
    sub.qr[10] = np.float32(1.4e-45)
    sub.qc[11] = np.float32(1.17549435e-38)
    cols.append(sub)
    thr = Column("edge_exact_thresholds", 95000.0, 280.0, 0.9)
    thr.qr[:] = 1.0e-9
    thr.nr[:] = 1.0e-2
    thr.qs[:] = 1.0e-9
    thr.qc[:] = 1.0e-15
    thr.nc[:] = 10.0
    thr.qi[:] = 1.0e-15
    thr.qr[20:25] = 1.0e-4
    thr.qs[20:25] = 1.0e-4
    thr.nr[20:25] = 1.0e3
    cols.append(thr)
    frz = Column("edge_exactly_t0c", 100000.0, 290.0, 0.95)
    frz.qs[:] = 3.0e-4
    frz.qr[:] = 2.0e-4
    frz.nr[:] = 3.0e3
    frz.qc[:] = 1.0e-4
    frz.nc[:] = 1.0e8
    frz.qi[:] = 1.0e-5
    for k in range(0, 12):
        frz.overrides[("pii", k)] = 1.0
        frz.overrides[("th", k)] = 273.15
    cols.append(frz)
    big = convective("edge_huge_drops", rng, 3.0, 2e8)
    big.nr[:] = np.where(big.qr > 1e-9, 1.0e-1, 0.0)
    cols.append(big)
    tiny = convective("edge_tiny_drops", rng, 1.0, 2e8)
    tiny.nr[:] = np.where(tiny.qr > 1e-9, 1.0e9, 0.0)
    tiny.nc[:] = np.where(tiny.qc > 1e-9, 5.0e10, 0.0)
    cols.append(tiny)
    sup = Column("edge_strong_supersaturation", 100000.0, 295.0, 1.10)
    sup.nn[:] = 1.5e9
    cols.append(sup)
    heavy = convective("edge_heavy_snow", rng, 5.0, 3e8)
    heavy.qs[:] = _blob(6000.0, 2500.0, 1.0e-2)
    cols.append(heavy)
    ice_only = Column("edge_ice_only_cold", 50000.0, 230.0, 1.2,
                      lapse=4.0e-3, ztrop=8000.0)
    ice_only.qi[:] = 2.0e-4
    cols.append(ice_only)
    drizzle = Column("edge_drizzle_into_cloud", 101000.0, 300.0, 1.0)
    drizzle.qc[:] = _blob(2000.0, 1000.0, 2e-3)
    drizzle.nc[:] = np.where(drizzle.qc > 1e-9, 5e7, 0.0)
    drizzle.qr[:] = _blob(1500.0, 300.0, 1e-5)
    drizzle.nr[:] = np.where(drizzle.qr > 1e-9, 1e5, 0.0)
    cols.append(drizzle)
    return cols


def assemble(cols):
    fields = [c.finish() for c in cols]
    levels = {name: np.stack([f[name] for f in fields], axis=1)
              for name in fields[0]}
    ncol = len(cols)
    surface = {"rain": np.linspace(0.0, 25.0, ncol).astype(np.float32),
               "snow": np.linspace(0.0, 3.0, ncol).astype(np.float32)}
    return levels, surface


def main(argv):
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    use_grid(NZ)
    cols = build_columns()
    levels, surface = assemble(cols)
    ncol = len(cols)
    for name, (delt, nsteps, it0) in CONFIGS.items():
        lv = {k: v.copy() for k, v in levels.items()}
        if it0 == 1:
            # Cold start: WRF overwrites nn with ccn0 at itimestep 1
            # (:216-224); WOOF fills nn with ccn0 at allocation.  Both
            # sides therefore start from the same field here.
            lv["nn"][...] = np.float32(CCN0)
        write_input(out / f"{name}.in",
                    Case(ncol, NZ, nsteps, it0, np.float32(delt),
                         np.float32(CCN0), lv, surface))
    use_grid(DEEP_NZ)
    deep_cols = build_columns()
    deep_levels, deep_surface = assemble(deep_cols)
    assert [c.name for c in deep_cols] == [c.name for c in cols]
    name, (delt, nsteps, it0) = DEEP_CONFIG
    write_input(out / f"{name}.in",
                Case(ncol, DEEP_NZ, nsteps, it0, np.float32(delt),
                     np.float32(CCN0), deep_levels, deep_surface))
    use_grid(NZ)
    with open(out / "columns.txt", "w", encoding="ascii") as f:
        for i, c in enumerate(cols):
            f.write(f"{i} {c.name}\n")
    print(f"{ncol} columns x {NZ} levels, {len(CONFIGS)} configurations, "
          f"plus {DEEP_CONFIG[0]} on {DEEP_NZ} levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
