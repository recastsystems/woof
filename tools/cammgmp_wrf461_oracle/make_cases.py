"""Input deck for the WRF v4.6.1 CAMMGMP (mp_physics = 11) column oracle.

Writes the binary deck run_cammgmp.F90 reads.  Every column is a physically
built sounding (hydrostatic interfaces, hypsometric heights, Goff-Gratch
saturation) with a regime laid over it; the regimes are the ones the brief
names -- deep convective, stratiform, mixed-phase, warm rain, dry and high
aerosol -- plus coupled deep/shallow convection (cu_physics = 7 /
shcu_physics = 2 inputs), a carried-state restart group (itimestep > 1 with
non-zero CLDFRA_OLD_MP / CLDFRA_MP_ALL) and an edge-case group (threshold
temperatures, supersaturation, negative and tiny moments, a 180 K top that
puts 373.16/T above 2, zero Aitken aerosol).

Determinism: numpy's PCG64 ``Generator.random`` draws (stable across numpy
releases), all arithmetic in float64, every array rounded once to float32.
The GPU harness (tests/test_cammgmp_wrf461_parity.py) reads this same deck.

usage: python make_cases.py OUT.bin [--npz OUT.npz]
"""

from __future__ import annotations

import argparse
import struct

import numpy as np

MAGIC = 1297040707          # b"CMGM" little-endian
VERSION = 1

#: The per-level arrays, in run_cammgmp.F90's READ order.  All are stored on
#: nz+1 levels (WRF's kms:kme); mass fields leave level nz+1 at zero.
FIELDS_3D = (
    "p8w", "p_hyd", "t_phy", "pi_phy", "z_at_w", "tke_pbl", "turbtype3d",
    "smaw3d", "dlf3d", "dlf2_3d", "z_sea_level", "kvh3d", "alt", "icwmrsh3d",
    "icwmrdp3d", "shfrc3d", "cmfmc3d", "cmfmc2_3d", "f_ice_phy", "f_rain_phy",
    "th", "cldfra_old_mp", "cldfra_mp", "cldfra_mp_all", "lradius", "iradius",
    "cldfrai", "cldfral", "cldfra_conv", "wsedl3d", "qv", "qc", "qi", "qs",
    "qr", "nc3d", "ni3d", "ns3d", "nr3d", "qndrop", "rh_old_mp", "lcd_old_mp",
    "dthdt", "dqvdt")
FIELDS_2D = ("qfx", "rliq2d", "ht", "xland", "snowh", "rainnc", "rainncv",
             "snownc", "snowncv", "sr")
#: What run_cammgmp.F90 writes after each step (mass levels 1..nz).
OUT_3D = ("th", "qv", "qc", "qi", "qs", "qr", "nc3d", "ni3d", "ns3d", "nr3d",
          "qndrop", "cldfra_old_mp", "cldfra_mp", "cldfra_mp_all", "lradius",
          "iradius", "cldfrai", "cldfral", "cldfra_conv", "wsedl3d",
          "rh_old_mp", "lcd_old_mp")
OUT_2D = ("rainnc", "rainncv", "snownc", "snowncv", "sr")

RD, CP, G, P0 = 287.0, 1004.5, 9.81, 1.0e5


def _esat_liq(t):
    """Goff-Gratch over liquid (Pa); only builds plausible soundings."""
    ts = 373.16
    x = (-7.90298 * (ts / t - 1.0) + 5.02808 * np.log10(ts / t)
         - 1.3816e-7 * (10.0 ** (11.344 * (1.0 - t / ts)) - 1.0)
         + 8.1328e-3 * (10.0 ** (-3.49149 * (ts / t - 1.0)) - 1.0)
         + np.log10(1013.246))
    return 100.0 * 10.0 ** x


def _esat_ice(t):
    t0 = 273.16
    x = (-9.09718 * (t0 / t - 1.0) - 3.56654 * np.log10(t0 / t)
         + 0.876793 * (1.0 - t / t0) + np.log10(6.1071))
    return 100.0 * 10.0 ** x


def _qsat(t, p, ice=False):
    es = np.where(ice & (t < 273.15), _esat_ice(t), _esat_liq(t))
    return 0.622 * es / np.maximum(p - 0.378 * es, 1.0)


class Column:
    """One sounding on nz mass levels / nz+1 interfaces, k = 1 at the surface."""

    def __init__(self, nz, rng, *, psfc=1.0e5, ptop=5.0e3, tsfc=295.0,
                 lapse=6.5e-3, ttrop=210.0, ht=0.0, ttop=None):
        self.nz = nz
        eta = np.linspace(0.0, 1.0, nz + 1) ** 1.25
        self.p8w = psfc - (psfc - ptop) * eta
        self.p_hyd = 0.5 * (self.p8w[:-1] + self.p8w[1:])
        # Hypsometric heights on a piecewise lapse-rate temperature profile.
        z = np.zeros(nz + 1)
        t_w = np.zeros(nz + 1)
        t_w[0] = tsfc
        for k in range(nz):
            tk = t_w[k]
            dz = RD * tk / G * np.log(self.p8w[k] / self.p8w[k + 1])
            z[k + 1] = z[k] + dz
            t_w[k + 1] = max(tsfc - lapse * z[k + 1], ttrop)
        if ttop is not None:
            t_w[-3:] = np.minimum(t_w[-3:], [ttop + 6.0, ttop + 3.0, ttop])
        self.z_at_w = z + ht
        self.ht = ht
        self.t = 0.5 * (t_w[:-1] + t_w[1:])
        self.z = 0.5 * (self.z_at_w[:-1] + self.z_at_w[1:])
        self.rng = rng
        self.f = {name: np.zeros(nz + 1) for name in FIELDS_3D}
        self.s = {name: 0.0 for name in FIELDS_2D}
        self.s["ht"] = ht
        self.s["xland"] = 1.0
        self.set_rh(np.full(nz, 0.6))

    def u(self, lo, hi, size=None):
        return lo + (hi - lo) * self.rng.random(size)

    def set_rh(self, rh, ice=False):
        nz = self.nz
        self.qv = rh * _qsat(self.t, self.p_hyd, ice=ice)
        self.qv = np.minimum(self.qv, 0.03)

    def layer(self, zlo, zhi):
        return (self.z - self.ht >= zlo) & (self.z - self.ht <= zhi)

    def finish(self):
        nz = self.nz
        f = self.f
        pi = (self.p_hyd / P0) ** (RD / CP)
        f["p8w"][:] = self.p8w
        f["p_hyd"][:nz] = self.p_hyd
        f["t_phy"][:nz] = self.t
        f["pi_phy"][:nz] = pi
        f["th"][:nz] = self.t / pi
        f["z_at_w"][:] = self.z_at_w
        f["z_sea_level"][:nz] = self.z
        f["qv"][:nz] = self.qv
        tv = self.t * (1.0 + 0.608 * self.qv)
        f["alt"][:nz] = RD * tv / self.p_hyd / (1.0 + self.qv)
        f["f_ice_phy"][:nz] = np.where(self.t < 253.15, 1.0, 0.0)
        return self


def _pbl(col, depth=1200.0, tke0=1.5, kvh0=60.0):
    """A boundary-layer TKE / K_h profile on interfaces (k = 1 surface)."""
    zw = col.z_at_w - col.ht
    shape = np.clip(1.0 - zw / depth, 0.0, 1.0) * np.clip(zw / 50.0 + 0.2, 0, 1)
    col.f["tke_pbl"][:] = tke0 * shape
    col.f["kvh3d"][:] = kvh0 * shape
    col.f["kvh3d"][0] = 0.0
    col.f["turbtype3d"][:] = np.where(shape > 0, 2.0, 0.0)
    col.f["smaw3d"][:] = np.where(shape > 0, 1.0, 0.0)


def _cloud(col, zlo, zhi, qc=0.0, qi=0.0, nc=0.0, ni=0.0):
    m = col.layer(zlo, zhi)
    nz = col.nz
    col.f["qc"][:nz][m] = qc * col.u(0.5, 1.5, m.sum())
    col.f["qi"][:nz][m] = qi * col.u(0.5, 1.5, m.sum())
    col.f["nc3d"][:nz][m] = nc * col.u(0.5, 1.5, m.sum())
    col.f["ni3d"][:nz][m] = ni * col.u(0.5, 1.5, m.sum())
    return m


def _precip_inputs(col):
    """qs/qr/ns/nr are OUTPUTS of CAMMGMP; nonzero inputs prove they are not read."""
    nz = col.nz
    col.f["qr"][:nz] = col.u(0.0, 2e-4, nz)
    col.f["qs"][:nz] = col.u(0.0, 2e-4, nz)
    col.f["nr3d"][:nz] = col.u(0.0, 5e3, nz)
    col.f["ns3d"][:nz] = col.u(0.0, 5e4, nz)
    col.f["qndrop"][:nz] = col.u(0.0, 1e8, nz)


def deep_convective(nz, rng, coupled):
    col = Column(nz, rng, tsfc=rng.uniform(298, 304), lapse=rng.uniform(6.0, 7.5))
    rh = np.where(col.z < 12000, rng.uniform(0.85, 1.02), 0.4)
    col.set_rh(rh)
    _cloud(col, 1500, 6000, qc=1.5e-3, nc=1.2e8)
    _cloud(col, 5000, 11000, qi=4e-4, ni=2e5)
    _pbl(col, depth=1500, tke0=2.5, kvh0=90)
    col.f["dthdt"][:nz] = np.where(col.layer(1000, 10000), -4e-4, 0.0)
    col.f["dqvdt"][:nz] = np.where(col.layer(1000, 6000), 2e-7, 0.0)
    if coupled:
        zw = col.z_at_w
        mf = np.clip(np.sin(np.pi * np.clip((zw - 500) / 11000, 0, 1)), 0, 1)
        col.f["cmfmc3d"][:] = rng.uniform(0.02, 0.08) * mf
        col.f["cmfmc2_3d"][:] = 0.005 * mf * (zw < 3000)
        col.f["icwmrdp3d"][:nz] = 1e-3 * mf[:-1]
        col.f["icwmrsh3d"][:nz] = 3e-4 * (col.z < 3000)
        col.f["shfrc3d"][:nz] = 0.05 * (col.z < 3000)
        col.f["dlf3d"][:nz] = 1e-7 * mf[:-1]
        col.f["dlf2_3d"][:nz] = 5e-8 * (col.z < 3000)
    _precip_inputs(col)
    return col.finish()


def shallow_cumulus(nz, rng):
    col = Column(nz, rng, tsfc=rng.uniform(292, 298))
    col.set_rh(np.where(col.z < 2500, 0.9, 0.5))
    _cloud(col, 800, 2200, qc=4e-4, nc=8e7)
    _pbl(col, depth=1000)
    zw = col.z_at_w
    col.f["cmfmc3d"][:] = 0.01 * ((zw > 500) & (zw < 2500))
    col.f["cmfmc2_3d"][:] = col.f["cmfmc3d"]
    col.f["icwmrsh3d"][:nz] = 5e-4 * ((col.z > 600) & (col.z < 2400))
    col.f["shfrc3d"][:nz] = rng.uniform(0.05, 0.2) * ((col.z > 600) & (col.z < 2400))
    col.f["dlf2_3d"][:nz] = 2e-8
    col.f["dthdt"][:nz] = -1e-4
    _precip_inputs(col)
    return col.finish()


def stratiform(nz, rng):
    col = Column(nz, rng, tsfc=rng.uniform(270, 285))
    rh = np.where(col.layer(1500, 6000), rng.uniform(0.95, 1.05), 0.7)
    col.set_rh(rh)
    _cloud(col, 2500, 5000, qc=2e-4, qi=5e-5, nc=6e7, ni=5e4)
    _cloud(col, 5000, 7000, qi=8e-5, ni=8e4)
    _pbl(col, depth=600, tke0=0.6, kvh0=20)
    col.f["dthdt"][:nz] = np.where(col.layer(1500, 6000), -1.5e-4, 0.0)
    _precip_inputs(col)
    return col.finish()


def mixed_phase(nz, rng):
    col = Column(nz, rng, tsfc=rng.uniform(262, 272), lapse=5.5e-3)
    col.set_rh(np.where(col.layer(0, 4000), rng.uniform(0.92, 1.03), 0.6))
    _cloud(col, 300, 3500, qc=3e-4, qi=1e-4, nc=5e7, ni=1e5)
    _pbl(col, depth=800, tke0=0.8)
    col.s["snowh"] = rng.uniform(0.0, 40.0)
    col.f["dthdt"][:nz] = -2e-4
    _precip_inputs(col)
    return col.finish()


def warm_rain(nz, rng):
    col = Column(nz, rng, tsfc=rng.uniform(296, 302))
    col.set_rh(np.where(col.z < 3500, rng.uniform(0.96, 1.01), 0.55))
    _cloud(col, 400, 3200, qc=rng.uniform(5e-4, 2.5e-3), nc=rng.uniform(3e7, 2e8))
    _pbl(col, depth=900, tke0=1.0, kvh0=40)
    col.s["xland"] = 2.0
    col.f["dthdt"][:nz] = np.where(col.z < 3500, -2e-4, 0.0)
    _precip_inputs(col)
    return col.finish()


def dry(nz, rng):
    col = Column(nz, rng, tsfc=rng.uniform(285, 310), lapse=rng.uniform(7e-3, 9.5e-3))
    col.set_rh(np.full(nz, rng.uniform(0.05, 0.35)))
    _pbl(col, depth=2500, tke0=3.0, kvh0=100)
    _precip_inputs(col)
    return col.finish()


def high_aerosol(nz, rng):
    kind = int(3 * rng.random())
    return (warm_rain, stratiform, mixed_phase)[int(kind)](nz, rng)


def edge_case(nz, rng, j):
    col = Column(nz, rng, tsfc=rng.uniform(270, 300),
                 ttop=180.0 if j % 4 == 0 else None, ht=1500.0 if j % 5 == 0 else 0.0,
                 psfc=8.4e4 if j % 5 == 0 else 1.0e5)
    nz_ = nz
    rh = np.full(nz_, 0.8)
    rh[2 + j % 5: 6 + j % 5] = 1.08          # supersaturated: findsp_water
    col.set_rh(rh)
    k0 = 3 + j % 7
    # Threshold temperatures: tmelt, -5 C, the Bergeron/rain-freezing 269.15,
    # 268.15/238.15 residual partition, -40 C homogeneous freezing.
    thresholds = (273.15, 268.15, 269.15, 238.15, 233.15, 270.16, 265.16, 275.15)
    col.t[k0] = thresholds[j % len(thresholds)]
    col.f["qc"][k0] = 2e-4
    col.f["qi"][k0] = 3e-5
    col.f["nc3d"][k0] = 4e7
    col.f["ni3d"][k0] = 2e4
    col.f["qc"][k0 + 1] = -1e-12              # negative tiny mass -> qmin reset
    col.f["qi"][k0 + 2] = 5e-12
    col.f["nc3d"][k0 + 3] = -3.0              # negative number -> qmin
    col.f["ni3d"][k0 + 4] = 1e-3
    col.f["qc"][k0 + 5] = 1e-17               # just above qsmall
    col.f["nc3d"][k0 + 5] = 1e11              # above the 1e10 number cap
    col.qv[k0 + 6] = -1e-9                    # negative vapour
    col.f["qi"][k0 + 7] = 6e-3                # in-cloud cap 5e-3
    col.f["ni3d"][k0 + 7] = 1.0
    _pbl(col, depth=500 + 200 * (j % 4), tke0=0.0 if j % 3 == 0 else 2.0,
         kvh0=500.0 if j % 6 == 1 else 10.0)
    col.s["snowh"] = 0.0005 if j % 2 else 5.0
    col.s["xland"] = 2.0 if j % 3 == 1 else 1.0
    col.f["dthdt"][:nz] = (j % 3 - 1) * 2e-4
    col.f["dqvdt"][:nz] = (j % 2) * 1e-7
    _precip_inputs(col)
    return col.finish()


def build_deck(seed: int = 20261008):
    rng = np.random.Generator(np.random.PCG64(seed))
    groups = []

    def add(cols, *, nstep, it0, cu, shcu, dt, accum=1000.0e6, aitken=300.0e6,
            coarse=0.2e6, carried=False):
        nz = cols[0].nz
        if carried:
            for col in cols:
                n = col.nz
                base = col.u(0.0, 0.9, n)
                col.f["cldfra_old_mp"][:n] = base
                col.f["cldfra_mp_all"][:n] = np.clip(base + col.u(-0.2, 0.2, n), 0, 1)
                col.f["cldfra_mp"][:n] = base
                col.f["rh_old_mp"][:n] = col.u(0.3, 1.0, n)
                col.f["lcd_old_mp"][:n] = base
                col.s["rainnc"] = col.u(0.0, 30.0)
                col.s["snownc"] = col.u(0.0, 5.0)
        groups.append(dict(cols=cols, nz=nz, nstep=nstep, it0=it0, cu=cu,
                           shcu=shcu, dt=dt, accum=accum, aitken=aitken,
                           coarse=coarse))

    nz = 44
    g1 = []
    for maker in (lambda: deep_convective(nz, rng, False), lambda: stratiform(nz, rng),
                  lambda: mixed_phase(nz, rng), lambda: warm_rain(nz, rng),
                  lambda: dry(nz, rng)):
        g1 += [maker() for _ in range(8)]
    add(g1, nstep=4, it0=1, cu=0, shcu=0, dt=60.0)
    g2 = [deep_convective(nz, rng, True) for _ in range(8)]
    g2 += [shallow_cumulus(nz, rng) for _ in range(8)]
    add(g2, nstep=4, it0=1, cu=7, shcu=2, dt=90.0)
    g3 = [high_aerosol(nz, rng) for _ in range(8)]
    add(g3, nstep=4, it0=1, cu=0, shcu=0, dt=120.0, accum=5.0e9, aitken=2.0e9,
        coarse=2.0e6)
    g4 = [(stratiform, mixed_phase, warm_rain, dry)[i % 4](nz, rng) for i in range(8)]
    add(g4, nstep=3, it0=37, cu=0, shcu=0, dt=60.0, carried=True)
    g5 = [edge_case(40, rng, j) for j in range(16)]
    add(g5, nstep=3, it0=1, cu=0, shcu=0, dt=30.0, accum=1.0e7, aitken=0.0,
        coarse=0.0)
    return groups


def deck_arrays(group):
    """float32 arrays (ncol, nz+1) / (ncol,) for one group, as written."""
    cols = group["cols"]
    a3 = {name: np.stack([c.f[name] for c in cols]).astype(np.float32)
          for name in FIELDS_3D}
    a2 = {name: np.array([c.s[name] for c in cols], dtype=np.float32)
          for name in FIELDS_2D}
    return a3, a2


def write_deck(path, groups):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<3i", MAGIC, VERSION, len(groups)))
        for g in groups:
            a3, a2 = deck_arrays(g)
            ncol = len(g["cols"])
            fh.write(struct.pack("<6i", ncol, g["nz"], g["nstep"], g["it0"],
                                 g["cu"], g["shcu"]))
            fh.write(struct.pack("<4f", g["dt"], g["accum"], g["aitken"],
                                 g["coarse"]))
            for name in FIELDS_3D:
                # Fortran (i, k) order: column fastest.
                fh.write(np.ascontiguousarray(a3[name].T).astype("<f4").tobytes())
            for name in FIELDS_2D:
                fh.write(a2[name].astype("<f4").tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    args = parser.parse_args()
    groups = build_deck()
    write_deck(args.out, groups)
    ncol = sum(len(g["cols"]) for g in groups)
    print(f"wrote {args.out}: {len(groups)} groups, {ncol} columns")


if __name__ == "__main__":
    main()
