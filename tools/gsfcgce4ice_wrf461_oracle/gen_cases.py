"""Generate the Goddard 4-ice column-oracle input files.

Three runs, one file each, every run the same 80 columns (column ``i`` keeps
its regime across runs) under a different clock:

  A  dt=18 s, dx=3 km,  itimestep 1..6, refl due on steps 1,3,4,6
  B  dt=60 s, dx=12 km, itimestep 10..13 with nonzero carried accumulators
  C  dt=6 s,  dx=1 km,  itimestep 2..4, do_radar_ref=0

The 80 columns: deep convection with hail (16), stratiform with a melting
layer (10), cold-season snow (10), warm rain (8), dry and stable (6), high
terrain (6), edge cases (16) and seeded random states (8).  Half of every
group is over land (XLAND=1), half over water (XLAND=2), so both CCN
branches of the cloud radius run.

Inputs are physical where a regime is named and deliberately unphysical
where an edge case needs it (signed zeros, subnormals, negative advected
residue, values exactly on the scheme's thresholds).  Every number written is
a float32 the oracle reads verbatim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import layout

NZ = 40
RD, CP, G = 287.0, 1004.0, 9.81
T0 = np.float32(273.16)       # consat_s t0
T00 = np.float32(238.16)      # consat_s t00


def z_faces(ht: float) -> np.ndarray:
    """NZ+1 interface heights (m MSL): 60 m at the ground to 900 m aloft."""
    dz = np.geomspace(60.0, 900.0, NZ)
    dz *= 19500.0 / dz.sum()
    return ht + np.concatenate([[0.0], np.cumsum(dz)])


def qsat_w(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def sounding(ht, t_sfc, lapse, trop_z, rh_low, rh_high, p_sfc=1.0e5):
    zw = z_faces(ht)
    zm = 0.5 * (zw[1:] + zw[:-1])
    dzw = zw[1:] - zw[:-1]
    agl = zm - ht
    t = np.where(agl < trop_z, t_sfc - lapse * agl,
                 t_sfc - lapse * trop_z)
    rh = np.interp(agl, [0.0, 3000.0, 8000.0, 20000.0],
                   [rh_low, rh_low, rh_high, 0.05])
    p = np.empty(NZ)
    pk = p_sfc * np.exp(-G * ht / (RD * t_sfc))
    for k in range(NZ):
        half = 0.5 * dzw[k]
        if k == 0:
            pk = pk * np.exp(-G * half / (RD * t[0]))
        else:
            tm = 0.5 * (t[k] + t[k - 1])
            pk = pk * np.exp(-G * (zm[k] - zm[k - 1]) / (RD * tm))
        p[k] = pk
    qv = rh * qsat_w(t, p)
    return zw, zm, dzw, agl, t, p, qv


def layer(agl, bottom, top, peak):
    """A smooth hydrometeor layer, zero outside [bottom, top]."""
    x = (agl - bottom) / max(top - bottom, 1.0)
    shape = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)), 0.0)
    return peak * shape ** 1.5


class Column:
    def __init__(self, label, ht, xland, t_sfc, lapse, trop_z, rh_low,
                 rh_high, p_sfc=1.0e5):
        self.label = label
        self.ht = ht
        self.xland = xland
        (self.zw, self.zm, self.dzw, self.agl, self.t, self.p,
         self.qv) = sounding(ht, t_sfc, lapse, trop_z, rh_low, rh_high, p_sfc)
        self.q = {n: np.zeros(NZ) for n in ("qc", "qr", "qi", "qs", "qg",
                                            "qh")}
        self.w = np.zeros(NZ + 1)

    def finish(self):
        """Float32 fields the scheme reads, derived as WRF's prep does."""
        p32 = self.p.astype(np.float32)
        pii = np.power(p32 / np.float32(1.0e5),
                       np.float32(RD / CP)).astype(np.float32)
        th = (self.t / pii.astype(np.float64)).astype(np.float32)
        tv = self.t * (1.0 + 0.608 * self.qv)
        rho = (self.p / (RD * tv)).astype(np.float32)
        zw32 = self.zw.astype(np.float32)
        z = (np.float32(0.5) * (zw32[1:] + zw32[:-1])).astype(np.float32)
        dz8w = (zw32[1:] - zw32[:-1]).astype(np.float32)
        out = {"th": th, "qv": self.qv.astype(np.float32), "rho": rho,
               "pii": pii, "p": p32, "z": z, "dz8w": dz8w}
        for n, v in self.q.items():
            out[n] = np.asarray(v, dtype=np.float32)
        return out, self.w.astype(np.float32)


def convective(i, rng):
    c = Column("deep-convective-hail", 0.0, 1.0 + (i % 2),
               t_sfc=rng.uniform(298, 306), lapse=rng.uniform(6.3, 7.2e0) / 1e3,
               trop_z=rng.uniform(12000, 14500), rh_low=rng.uniform(0.85, 1.0),
               rh_high=rng.uniform(0.7, 1.02))
    a = c.agl
    s = rng.uniform(0.5, 1.5)
    c.q["qc"] = layer(a, 800, 9000, 2.5e-3 * s)
    c.q["qr"] = layer(a, 0, 6000, 4.0e-3 * s)
    c.q["qi"] = layer(a, 7000, 14000, 6.0e-4 * s)
    c.q["qs"] = layer(a, 5000, 13000, 1.2e-3 * s)
    c.q["qg"] = layer(a, 3000, 11000, 3.0e-3 * s)
    c.q["qh"] = layer(a, 1500, 10500, rng.choice([5e-4, 3e-3, 1.2e-2]) * s)
    sign = -1.0 if i % 4 == 3 else 1.0
    c.w = sign * layer(c.zw - c.ht, 0, 15000, rng.uniform(8, 35))
    return c


def stratiform(i, rng):
    c = Column("stratiform-melting", 0.0, 1.0 + (i % 2),
               t_sfc=rng.uniform(285, 292), lapse=6.0e-3, trop_z=11000,
               rh_low=rng.uniform(0.85, 1.0), rh_high=rng.uniform(0.9, 1.05))
    a = c.agl
    melt = (c.t[0] - 273.15) / 6.0e-3
    c.q["qs"] = layer(a, melt - 300, melt + 6000, rng.uniform(1e-4, 1.2e-3))
    c.q["qr"] = layer(a, 0, melt + 200, rng.uniform(5e-5, 6e-4))
    c.q["qi"] = layer(a, melt + 3000, melt + 8000, 1.5e-4)
    c.q["qg"] = layer(a, melt - 200, melt + 1500, rng.uniform(0, 2e-4))
    c.q["qc"] = layer(a, melt - 800, melt + 1000, rng.uniform(0, 2e-4))
    c.w = layer(c.zw - c.ht, 1000, 9000, rng.uniform(-0.4, 0.6))
    return c


def cold_snow(i, rng):
    c = Column("cold-season-snow", 0.0, 1.0 + (i % 2),
               t_sfc=rng.uniform(258, 271), lapse=5.5e-3, trop_z=9000,
               rh_low=rng.uniform(0.85, 1.0), rh_high=rng.uniform(0.8, 1.1))
    a = c.agl
    c.q["qs"] = layer(a, 0, 6000, rng.uniform(5e-5, 8e-4))
    c.q["qi"] = layer(a, 1000, 8000, rng.uniform(1e-5, 3e-4))
    c.q["qc"] = layer(a, 0, 1500, rng.uniform(0, 3e-4))
    c.q["qg"] = layer(a, 0, 2500, rng.uniform(0, 1e-4))
    c.w = layer(c.zw - c.ht, 0, 6000, rng.uniform(-0.2, 0.5))
    return c


def warm_rain(i, rng):
    c = Column("warm-rain", 0.0, 1.0 + (i % 2),
               t_sfc=rng.uniform(296, 302), lapse=5.0e-3, trop_z=15000,
               rh_low=rng.uniform(0.9, 1.03), rh_high=0.6)
    a = c.agl
    c.q["qc"] = layer(a, 500, 3500, rng.uniform(2e-4, 2e-3))
    c.q["qr"] = layer(a, 0, 3000, rng.uniform(1e-5, 2e-3))
    c.w = layer(c.zw - c.ht, 0, 4000, rng.uniform(-1.0, 3.0))
    return c


def dry_stable(i, rng):
    c = Column("dry-stable", 0.0, 1.0 + (i % 2),
               t_sfc=rng.uniform(270, 300), lapse=rng.uniform(2e-3, 5e-3),
               trop_z=11000, rh_low=rng.uniform(0.05, 0.3),
               rh_high=rng.uniform(0.02, 0.2))
    # A surface inversion: the lowest kilometre warms with height.
    inv = c.agl < 1000.0
    c.t = np.where(inv, c.t + 6.0 * c.agl / 1000.0, c.t + 6.0)
    c.qv = 0.2 * qsat_w(c.t, c.p)
    c.w = layer(c.zw - c.ht, 0, 12000, rng.uniform(-0.05, 0.05))
    return c


def high_terrain(i, rng):
    ht = rng.uniform(2500, 4200)
    c = Column("high-terrain", ht, 1.0 + (i % 2),
               t_sfc=rng.uniform(268, 290), lapse=6.5e-3, trop_z=10000,
               rh_low=rng.uniform(0.7, 1.0), rh_high=rng.uniform(0.6, 1.0))
    a = c.agl
    c.q["qs"] = layer(a, 0, 5000, rng.uniform(1e-4, 1e-3))
    c.q["qg"] = layer(a, 0, 4000, rng.uniform(1e-4, 2e-3))
    c.q["qh"] = layer(a, 0, 3000, rng.uniform(0, 2e-3))
    c.q["qr"] = layer(a, 0, 1500, rng.uniform(0, 5e-4))
    c.q["qc"] = layer(a, 200, 3000, rng.uniform(0, 5e-4))
    c.q["qi"] = layer(a, 3000, 8000, rng.uniform(0, 2e-4))
    c.w = layer(c.zw - c.ht, 0, 8000, rng.uniform(-3, 8))
    return c


def random_state(i, rng):
    c = Column("seeded-random", rng.uniform(0, 1500), 1.0 + (i % 2),
               t_sfc=rng.uniform(260, 305), lapse=rng.uniform(4e-3, 8e-3),
               trop_z=rng.uniform(9000, 15000), rh_low=rng.uniform(0.3, 1.05),
               rh_high=rng.uniform(0.3, 1.05))
    for n, top in (("qc", 3e-3), ("qr", 5e-3), ("qi", 1e-3), ("qs", 3e-3),
                   ("qg", 6e-3), ("qh", 1.5e-2)):
        mask = rng.random(NZ) < 0.6
        c.q[n] = np.where(mask, top * rng.random(NZ) ** 3, 0.0)
    c.w = rng.normal(0.0, 4.0, NZ + 1)
    return c


def edge(i, rng):
    """Threshold and representation edges; each column sets several."""
    base = convective(i, rng) if i % 2 == 0 else cold_snow(i, rng)
    base.label = "edge-cases"
    return base


def edge_finish(i, fields, w):
    """Overwrite words of an edge column with exact threshold values."""
    f32 = np.float32
    k = 3 + (i % 5)
    if i % 8 == 0:   # signed zeros and subnormals everywhere they can sit
        for n in ("qc", "qr", "qi", "qs", "qg", "qh"):
            fields[n][k] = f32(-0.0)
            fields[n][k + 1] = np.float32(1.0e-40)
            fields[n][k + 2] = np.float32(-1.0e-40)
            fields[n][k + 3] = np.float32(1.4e-45)
        w[k] = f32(-0.0)
    if i % 8 == 1:   # exactly cmin and its neighbours, and negative residue
        for n in ("qc", "qr", "qi", "qs", "qg", "qh"):
            fields[n][k] = f32(1.0e-20)
            fields[n][k + 1] = np.nextafter(f32(1.0e-20), f32(1.0))
            fields[n][k + 2] = np.nextafter(f32(1.0e-20), f32(0.0))
            fields[n][k + 3] = f32(-1.0e-9)
    if i % 8 == 2:   # the bulk-density switch rho*qg == qrog2 (2e-6 g cm-3)
        rho_cgs = fields["rho"] * f32(0.001)
        for kk in range(k, k + 4):
            q = f32(2.0e-6) / rho_cgs[kk]
            fields["qg"][kk] = q
            fields["qh"][kk] = q
            fields["qs"][kk] = q
        fields["qg"][k + 1] = np.nextafter(fields["qg"][k + 1], f32(1.0))
    if i % 8 == 3:   # tair exactly t0, t00, t0-5 and 268.16 at four levels
        for kk, target in zip(range(k, k + 4),
                              (T0, T00, f32(268.16), f32(273.16 - 5.0))):
            pii = fields["pii"][kk]
            th = f32(target / pii)
            for _ in range(64):
                if f32(th * pii) == target:
                    break
                th = np.nextafter(th, f32(1e9) if f32(th * pii) < target
                                  else f32(0.0))
            fields["th"][kk] = th
            fields["qc"][kk] = f32(3.0e-4)
            fields["qi"][kk] = f32(2.0e-5)
    if i % 8 == 4:   # dlt thresholds: qr 1e-4, qc 5e-4, qs 1e-4, qc+qi 1e-5
        fields["qr"][k] = f32(1.0e-4)
        fields["qc"][k] = f32(5.0e-4)
        fields["qs"][k] = f32(1.0e-4)
        fields["qc"][k + 1] = f32(0.5e-5)
        fields["qi"][k + 1] = f32(0.5e-5)
        fields["qr"][k + 2] = f32(0.001)
    if i % 8 == 5:   # supersaturation well past the ice cap, w on 0 and -2
        fields["qv"][k:k + 6] *= f32(1.25)
        w[k] = f32(0.0)
        w[k + 1] = f32(2.0)
        w[k + 2] = f32(-2.0)
    if i % 8 == 6:   # enormous hail and graupel at the top of the column
        fields["qh"][-6:] = f32(3.0e-2)
        fields["qg"][-6:] = f32(2.0e-2)
        fields["qr"][-3:] = f32(1.0e-2)
    if i % 8 == 7:   # one isolated word of each species at the lowest level
        for n in ("qc", "qr", "qi", "qs", "qg", "qh"):
            fields[n][:] = f32(0.0)
            fields[n][0] = f32(1.0e-3)


GROUPS = ((convective, 16), (stratiform, 10), (cold_snow, 10),
          (warm_rain, 8), (dry_stable, 6), (high_terrain, 6), (edge, 16),
          (random_state, 8))


def build_columns(seed: int = 20261008):
    rng = np.random.default_rng(seed)
    cols = []
    for maker, count in GROUPS:
        for j in range(count):
            c = maker(j, rng)
            fields, w = c.finish()
            if maker is edge:
                edge_finish(j, fields, w)
            cols.append((c, fields, w))
    return cols


def assemble(cols, *, dt, dx, itimestep, diagflag, do_radar_ref,
             carried, rng):
    ncol = len(cols)
    level = {n: np.zeros((NZ, ncol), np.float32) for n in layout.LEVEL_FIELDS}
    w = np.zeros((NZ + 1, ncol), np.float32)
    ht = np.zeros(ncol, np.float32)
    xland = np.zeros(ncol, np.float32)
    for i, (c, fields, wc) in enumerate(cols):
        for n, v in fields.items():
            level[n][:, i] = v
        w[:, i] = wc
        ht[i] = c.ht
        xland[i] = c.xland
    surface = {n: np.zeros(ncol, np.float32) for n in layout.SURFACE_FIELDS}
    if carried:
        # Nonzero carried accumulators and diagnostics: the INOUT fields
        # must be updated from these values, not from zero.
        for n in ("acphysc", "acphyse", "acphysd", "acphyss", "acphysm",
                  "acphysf"):
            level[n] = rng.uniform(-0.5, 3.0, (NZ, ncol)).astype(np.float32)
        for n in ("physc", "physe", "physd", "physs", "physm", "physf",
                  "re_cloud_gsfc", "re_rain_gsfc", "re_ice_gsfc",
                  "re_snow_gsfc", "re_graupel_gsfc", "re_hail_gsfc"):
            level[n] = rng.uniform(0.0, 50.0, (NZ, ncol)).astype(np.float32)
        level["refl_10cm"] = rng.uniform(-35.0, 60.0,
                                         (NZ, ncol)).astype(np.float32)
        for n in ("rainnc", "snownc", "graupelnc", "hailnc"):
            surface[n] = rng.uniform(0.0, 40.0, ncol).astype(np.float32)
        for n in ("rainncv", "snowncv", "graupelncv", "hailncv", "sr"):
            surface[n] = rng.uniform(0.0, 1.0, ncol).astype(np.float32)
    else:
        level["refl_10cm"][...] = np.float32(-35.0)
    return layout.Case(np.float32(dt), np.float32(dx),
                       np.asarray(itimestep, np.int32),
                       np.asarray(diagflag, np.int32), int(do_radar_ref),
                       ht, xland, level, w, surface,
                       [c.label for c, _, _ in cols])


RUNS = {
    "A": dict(dt=18.0, dx=3000.0, itimestep=[1, 2, 3, 4, 5, 6],
              diagflag=[1, 0, 1, 1, 0, 1], do_radar_ref=1, carried=False),
    "B": dict(dt=60.0, dx=12000.0, itimestep=[10, 11, 12, 13],
              diagflag=[0, 1, 0, 1], do_radar_ref=1, carried=True),
    "C": dict(dt=6.0, dx=1000.0, itimestep=[2, 3, 4],
              diagflag=[1, 1, 1], do_radar_ref=0, carried=True),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = build_columns()
    labels = [c.label for c, _, _ in cols]
    for name, spec in RUNS.items():
        case = assemble(cols, rng=np.random.default_rng(ord(name)), **spec)
        layout.write_input(case, args.out_dir / f"case-{name}.bin")
    (args.out_dir / "columns.txt").write_text(
        "".join(f"{i} {lab}\n" for i, lab in enumerate(labels)))
    print(f"wrote {len(RUNS)} runs x {len(cols)} columns x {NZ} levels "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
