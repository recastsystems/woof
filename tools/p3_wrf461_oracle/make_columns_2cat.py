#!/usr/bin/env python3
"""Column fixtures for the P3 two-category (mp_physics=52) oracle.

The byte format is evidence/p3-fortran-oracle-20260828/FORMAT.md widened to
the mp=52 call shape (module_microphysics_driver.F:1654-1706: two ice
categories and the predicted droplet number ``nc_3d``), read by
``p3_ref_driver_2cat.f90`` and by ``run_woof.py`` from the SAME BYTES:

  twenty (ncol, nk) inputs   th qv qc qr nr nc qi1 ni1 qir1 qib1
                             qi2 ni2 qir2 qib2 th_old qv_old pii p dz w
  five (ncol,) inputs        rainnc rainncv sr snownc snowncv

The soundings come from make_columns.py (the mp=50 oracle's own analytic
generator, unchanged); this file only adds the second category, the droplet
number and the columns aimed at what nCat > 1 adds to p3_main:

* ``interact``  both categories at the same levels, low rime fraction, two
  very different mean sizes: the ice-ice collection between categories in
  both directions (module_mp_p3.F:2754-2829, table 2) and its sources and
  sinks in the conservation and update passes (:3609-3630, :3801-3827).
* ``merge``     two categories with nearly the same mean diameter: the
  category merge (:4615-4697) fires when |Di(2) - Di(1)| <= 500 um.
* ``riming``    graupel in category 1, snow in category 2, supercooled
  cloud: rime density per category and wet growth.
* ``depnuc``    ice supersaturation below -15 C with large ice in category 1
  only: new ice goes to the EMPTY category (icecat_destination case 3b).
* ``splinter`` rain on ice whose mean-mass diameter exceeds 4 mm between
  265 and 270 K: Hallett-Mossop rime splintering, which nCat > 1 switches
  on (log_hmossopOn, :2256) and routes by icecat_destination.
* ``homfrz``    supercooled cloud and rain below 233.15 K beside two
  categories of different size: homogeneous-freezing destination.
* ``melt``      both categories melting.
* ``cat2only``  category 2 populated, category 1 empty.
* ``split``     every mp=50 regime column with a second ice population added
  (different size, rime and layer), so the whole mp=50 regime set runs
  through the nCat = 2 code.

Nothing here calls P3 or WOOF code.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_columns as M  # noqa: E402  (the mp=50 generator, reused)

NK = M.NK
IN3D = ["th", "qv", "qc", "qr", "nr", "nc",
        "qi1", "ni1", "qir1", "qib1", "qi2", "ni2", "qir2", "qib2",
        "th_old", "qv_old", "pii", "p", "dz", "w"]
IN2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]
OUT3D = ["th", "qv", "qc", "qr", "nr", "nc",
         "qi1", "ni1", "qir1", "qib1", "qi2", "ni2", "qir2", "qib2",
         "th_old", "qv_old",
         "zdbz", "effc", "effi", "vmi", "di", "rhopo", "vmi2", "di2", "rhopo2"]
OUT2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]
U = M.U


class Column2(M.Column):
    """An mp=50 sounding plus a second ice category and droplet number."""

    def __init__(self, label, regime, **kw):
        super().__init__(label, regime, **kw)
        self.h2 = {n: np.zeros(NK) for n in ("qi", "ni", "qir", "qib")}
        self.rdrop = 10.0e-6

    @classmethod
    def from_column(cls, c, label=None):
        new = cls.__new__(cls)
        new.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else
                                 ({n: a.copy() for n, a in v.items()}
                                  if isinstance(v, dict) else v))
                             for k, v in c.__dict__.items()})
        new.h2 = {n: np.zeros(NK) for n in ("qi", "ni", "qir", "qib")}
        new.rdrop = 10.0e-6
        if label:
            new.label = label
        return new

    def add_ice2(self, amount, zbot, ztop, *, mmean=1.0e-9, rimefrac=0.0,
                 rhorime=400.0):
        s = amount * self.layer(zbot, ztop)
        self.h2["qi"] += s
        self.h2["ni"] += np.where(s > 0, s / mmean, 0.0)
        self.h2["qir"] += rimefrac * s
        self.h2["qib"] += rimefrac * s / rhorime

    def fields(self):
        f = super().fields()
        out = {"th": f["th"], "qv": f["qv"], "qc": f["qc"], "qr": f["qr"],
               "nr": f["nr"]}
        m = np.pi / 6.0 * 1000.0 * (2.0 * self.rdrop) ** 3
        out["nc"] = np.where(f["qc"] > 0.0, f["qc"] / m, 0.0)
        for n in ("qi", "ni", "qir", "qib"):
            out[n + "1"] = f[n]
            out[n + "2"] = self.h2[n]
        for n in ("th_old", "qv_old", "pii", "p", "dz", "w"):
            out[n] = f[n]
        return out


def _cold(label, regime, rng, *, tsfc, lapse=6.5e-3, psfc=None):
    c = Column2(label, regime, psfc=psfc or U(rng, 97000, 102000),
                tsfc=tsfc, lapse=lapse, ztrop=U(rng, 9000, 12000))
    c.rdrop = U(rng, 6e-6, 14e-6)
    return c


def interact(rng, n=8):
    out = []
    for j in range(n):
        c = _cold(f"interact-{j:02d}", "two-cat-interaction", rng,
                  tsfc=U(rng, 266, 276))
        top = U(rng, 4000, 8000)
        c.rh_mixed(np.where(c.z < top, U(rng, 0.98, 1.08), 0.6))
        c.add_ice(U(rng, 0.2e-3, 2e-3), 0.0, top,
                  mmean=10 ** U(rng, -8.5, -6.5),
                  rimefrac=U(rng, 0.0, 0.55), rhorime=U(rng, 100, 500))
        c.add_ice2(U(rng, 0.05e-3, 1e-3), U(rng, 0, 1500), top,
                   mmean=10 ** U(rng, -12, -9.5),
                   rimefrac=U(rng, 0.0, 0.3), rhorime=U(rng, 100, 400))
        if j % 2 == 0:
            c.add_cloud(U(rng, 0.02e-3, 0.3e-3), U(rng, 500, 1500),
                        U(rng, 2000, 3500))
        c.tendency(rng, dth=0.05)
        out.append(c)
    return out


def merge(rng, n=6):
    out = []
    for j in range(n):
        c = _cold(f"merge-{j:02d}", "two-cat-merge", rng,
                  tsfc=U(rng, 255, 270))
        top = U(rng, 3000, 6000)
        c.rh_mixed(np.where(c.z < top, U(rng, 0.95, 1.05), 0.5))
        mm = 10 ** U(rng, -9.5, -7.5)
        fr = U(rng, 0.0, 0.5)
        rr = U(rng, 200, 600)
        a = U(rng, 0.1e-3, 1e-3)
        c.add_ice(a, 0.0, top, mmean=mm, rimefrac=fr, rhorime=rr)
        # the same population, rescaled in mass by up to 30 %, so the two
        # mean diameters sit inside deltaD_init = 500 um
        c.add_ice2(a * U(rng, 0.3, 1.3), 0.0, top,
                   mmean=mm * U(rng, 0.8, 1.25), rimefrac=fr, rhorime=rr)
        c.tendency(rng, dth=0.03)
        out.append(c)
    return out


def riming(rng, n=6):
    out = []
    for j in range(n):
        c = _cold(f"riming-{j:02d}", "two-cat-riming", rng,
                  tsfc=U(rng, 270, 280))
        zb, zt = U(rng, 500, 1500), U(rng, 3000, 6000)
        c.rh_mixed(np.where((c.z > zb) & (c.z < zt), U(rng, 1.0, 1.03), 0.8))
        c.add_cloud(U(rng, 0.2e-3, 2e-3), zb, zt)
        c.add_ice(U(rng, 0.3e-3, 3e-3), zb, zt + 2000,
                  mmean=10 ** U(rng, -8, -6), rimefrac=U(rng, 0.5, 0.95),
                  rhorime=U(rng, 400, 900))
        c.add_ice2(U(rng, 0.1e-3, 1e-3), zb, zt + 3000,
                   mmean=10 ** U(rng, -10.5, -8.5),
                   rimefrac=U(rng, 0.0, 0.2), rhorime=U(rng, 50, 300))
        if j % 2 == 1:
            c.add_rain(U(rng, 0.05e-3, 1e-3), 0.0, zt,
                       dmean=U(rng, 0.4e-3, 1.5e-3))
        c.tendency(rng, dth=0.1, dqv_rel=4e-3)
        out.append(c)
    return out


def depnuc(rng, n=6):
    out = []
    for j in range(n):
        c = _cold(f"depnuc-{j:02d}", "two-cat-deposition-nucleation", rng,
                  tsfc=U(rng, 250, 262), lapse=U(rng, 6e-3, 8e-3))
        c.rh_mixed(np.where(c.z > 1000, U(rng, 1.08, 1.30), 0.9))
        # large aggregates in category 1 only: new ice (D ~ 20 um) is more
        # than deltaD_init from them, so it starts category 2
        c.add_ice(U(rng, 0.1e-3, 1e-3), 0.0, U(rng, 4000, 8000),
                  mmean=10 ** U(rng, -7.5, -6), rimefrac=U(rng, 0, 0.3))
        if j % 3 == 2:   # and a few columns where category 2 already exists
            c.add_ice2(U(rng, 1e-6, 5e-5), 2000, 9000, mmean=1e-12)
        c.tendency(rng, dth=0.1)
        out.append(c)
    return out


def splinter(rng, n=4):
    out = []
    for j in range(n):
        c = _cold(f"splinter-{j:02d}", "two-cat-rime-splintering", rng,
                  tsfc=U(rng, 274, 280), lapse=U(rng, 5.5e-3, 6.5e-3))
        c.rh_mixed(np.full(NK, U(rng, 0.98, 1.02)))
        c.add_rain(U(rng, 0.5e-3, 3e-3), 0.0, 4000,
                   dmean=U(rng, 0.8e-3, 2e-3))
        c.add_cloud(U(rng, 0.2e-3, 1e-3), 500, 4000)
        # mean-mass diameter ((6 q/N)/(500 pi))**(1/3) above 4 mm: q/N above
        # 1.7e-5 kg (module_mp_p3.F:2578, :3064)
        c.add_ice(U(rng, 1e-3, 4e-3), 0.0, 5000, mmean=10 ** U(rng, -4.6, -4.2),
                  rimefrac=U(rng, 0.6, 0.95), rhorime=U(rng, 500, 900))
        if j % 2 == 0:
            c.add_ice2(U(rng, 0.05e-3, 0.5e-3), 1000, 6000,
                       mmean=10 ** U(rng, -10, -8.5))
        c.tendency(rng, dth=0.05)
        out.append(c)
    return out


def homfrz(rng, n=4):
    out = []
    for j in range(n):
        c = Column2(f"homfrz-{j:02d}", "two-cat-homogeneous-freezing",
                    psfc=U(rng, 99000, 101000), tsfc=U(rng, 298, 304),
                    lapse=U(rng, 7.0e-3, 7.8e-3), ztrop=U(rng, 14500, 16500))
        c.rdrop = U(rng, 5e-6, 12e-6)
        c.rh_mixed(np.where((c.z > 9000) & (c.z < 14500), U(rng, 1.0, 1.2),
                            U(rng, 0.5, 0.9)))
        c.add_ice(U(rng, 0.1e-3, 1e-3), 9000, 14500,
                  mmean=10 ** U(rng, -8.5, -7), rimefrac=U(rng, 0.0, 0.4))
        if j % 2 == 0:
            c.add_ice2(U(rng, 0.05e-3, 0.5e-3), 9000, 14500,
                       mmean=10 ** U(rng, -12.5, -11))
        c.add_cloud(U(rng, 0.05e-3, 1e-3), 10000, 12500)
        c.add_rain(U(rng, 0.05e-3, 0.8e-3), 10000, 12500,
                   dmean=U(rng, 0.3e-3, 1e-3))
        c.tendency(rng, dth=0.3, dqv_rel=1e-2)
        out.append(c)
    return out


def melt(rng, n=4):
    out = []
    for j in range(n):
        c = _cold(f"melt-{j:02d}", "two-cat-melting", rng,
                  tsfc=U(rng, 279, 290), lapse=U(rng, 5.5e-3, 6.5e-3))
        zfrz = (c.T[0] - 273.15) / 6.0e-3
        c.rh_mixed(np.full(NK, U(rng, 0.85, 1.0)))
        c.add_ice(U(rng, 0.2e-3, 2e-3), 0.0, zfrz + 3000,
                  mmean=10 ** U(rng, -8, -6), rimefrac=U(rng, 0.3, 0.9),
                  rhorime=U(rng, 300, 900))
        c.add_ice2(U(rng, 0.05e-3, 1e-3), 0.0, zfrz + 4000,
                   mmean=10 ** U(rng, -10, -8), rimefrac=U(rng, 0.0, 0.2))
        c.add_rain(U(rng, 0.05e-3, 1e-3), 0.0, zfrz)
        c.tendency(rng, dth=0.05)
        out.append(c)
    return out


def cat2only(rng, n=4):
    out = []
    for j in range(n):
        c = _cold(f"cat2only-{j:02d}", "two-cat-category-2-only", rng,
                  tsfc=U(rng, 255, 272))
        top = U(rng, 3000, 7000)
        c.rh_mixed(np.where(c.z < top, U(rng, 1.0, 1.15), 0.6))
        c.add_ice2(U(rng, 0.05e-3, 1e-3), 0.0, top,
                   mmean=10 ** U(rng, -10, -7), rimefrac=U(rng, 0, 0.6),
                   rhorime=U(rng, 100, 700))
        if j % 2 == 0:
            c.add_cloud(U(rng, 0.05e-3, 0.4e-3), 500, 2500)
        c.tendency(rng, dth=0.05)
        out.append(c)
    return out


def split_regimes(rng):
    """Every mp=50 regime column, with a second ice population added."""
    out = []
    for c in M.regime_columns():
        c2 = Column2.from_column(c, label="split-" + c.label)
        c2.rdrop = U(rng, 6e-6, 14e-6)
        cold = c2.z[c2.T < 268.0]
        if cold.size:
            zb = float(cold.min())
            c2.add_ice2(U(rng, 0.02e-3, 1e-3), zb, zb + U(rng, 2000, 6000),
                        mmean=10 ** U(rng, -12, -7),
                        rimefrac=U(rng, 0.0, 0.7), rhorime=U(rng, 50, 900))
        out.append(c2)
    return out


def edges(rng):
    out = []

    def base(label, **kw):
        args = dict(psfc=100000.0, tsfc=265.0, lapse=6.5e-3)
        args.update(kw)
        c = Column2(label, "edge", **args)
        c.rh_mixed(np.full(NK, 0.95))
        c.tendency(rng, dth=0.05)
        return c

    QS = 1.0e-14
    c = base("edge2-qsmall-straddle")
    c.h["qi"][5:15] = QS
    c.h["ni"][5:15] = 1.0e-3
    c.h2["qi"][5:15] = QS * 1.1
    c.h2["ni"][5:15] = 1.0e-3
    c.h2["qi"][20:30] = QS * 0.9
    out.append(c)

    c = base("edge2-ni2-zero")
    c.h["qi"][0:20] = 1.0e-4
    c.h["ni"][0:20] = 1.0e4
    c.h2["qi"][0:25] = 1.0e-4          # ni2 left 0
    c.rh_mixed(np.full(NK, 1.02))
    out.append(c)

    c = base("edge2-total-ni-cap", tsfc=250.0)
    c.h["qi"][0:30] = 5.0e-4
    c.h["ni"][0:30] = 3.0e6
    c.h2["qi"][0:30] = 2.0e-4
    c.h2["ni"][0:30] = 5.0e6           # sum far above max_total_Ni
    out.append(c)

    c = base("edge2-rime-inconsistent")
    c.h2["qi"][0:20] = 3.0e-4
    c.h2["ni"][0:20] = 1.0e5
    c.h2["qir"][0:10] = 6.0e-4
    c.h2["qib"][0:10] = 6.0e-4 / 500.0
    c.h2["qir"][10:20] = 1.0e-4
    c.h["qi"][5:25] = 1.0e-4
    c.h["ni"][5:25] = 1.0e3
    out.append(c)

    c = base("edge2-small-negatives", tsfc=275.0)
    for n in ("qi", "ni"):
        c.h2[n][3:30:3] = -1.0e-10
        c.h[n][4:30:3] = -1.0e-10
    c.h["qc"][10:20] = 2.0e-4
    out.append(c)

    c = base("edge2-identical-categories", tsfc=262.0)
    for h in (c.h, c.h2):
        h["qi"][0:25] = 4.0e-4
        h["ni"][0:25] = 2.0e4
        h["qir"][0:25] = 1.0e-4
        h["qib"][0:25] = 1.0e-4 / 400.0
    out.append(c)

    c = base("edge2-threshold-temperatures", tsfc=290.0)
    for target in (273.15, 269.15, 258.15, 233.15):
        k = int(np.argmin(np.abs(c.T - target)))
        c.T[k] = target
    c.rh_mixed(np.full(NK, 1.0))
    c.add_cloud(3.0e-4, 0, 12000)
    c.add_rain(3.0e-4, 0, 12000)
    c.add_ice(3.0e-4, 0, 12000, rimefrac=0.5, rhorime=400.0)
    c.add_ice2(1.0e-4, 0, 12000, mmean=1e-11)
    out.append(c)

    c = base("edge2-rimed-collectee", tsfc=271.0)
    c.rh_mixed(np.full(NK, 1.0))
    c.add_ice(1e-3, 0, 4000, mmean=1e-7, rimefrac=0.95, rhorime=900)
    c.add_ice2(1e-3, 0, 4000, mmean=1e-10, rimefrac=0.92, rhorime=850)
    out.append(c)
    return out


def targeted(rng):
    cols = []
    for maker in (interact, merge, riming, depnuc, splinter, homfrz, melt,
                  cat2only):
        cols += maker(rng)
    return cols


#: Columns of T30 committed with their WRF dump (tests/data/oracles/p3_2cat).
SUBSET_LABELS = ("interact-00", "interact-03", "merge-00", "riming-01",
                 "depnuc-02", "splinter-00", "homfrz-00", "melt-00",
                 "cat2only-00", "split-conv-00")


def pack(cols):
    ncol = len(cols)
    fs = [c.fields() for c in cols]
    blob = b""
    for name in IN3D:
        a = np.stack([f[name] for f in fs])
        blob += np.asarray(a, dtype=np.float64).astype("<f4").tobytes(
            order="F")
    for name in IN2D:
        blob += np.zeros(ncol, dtype="<f4").tobytes()
    return blob


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20261008)
    tcols = targeted(rng)
    scols = split_regimes(np.random.default_rng(52))
    ecols = edges(np.random.default_rng(5252))
    allc = tcols + scols
    by = {c.label: c for c in allc}
    fixtures = [
        ("T01-twocat-1step", allc, 20.0, 1, False),
        ("T30-twocat-30step", allc, 20.0, 30, False),
        ("T60-twocat-dt60", allc, 60.0, 10, False),
        ("E05-twocat-edges-5step", ecols, 20.0, 5, False),
        ("S30-twocat-subset-30step", [by[x] for x in SUBSET_LABELS], 20.0,
         30, False),
        ("X01-twocat-coldstart", tcols[:6] + ecols[:2], 20.0, 2, True),
    ]
    index = []
    for name, cols, dt, nsteps, cold in fixtures:
        blob = pack(cols)
        if cold:
            buf = np.frombuffer(blob, dtype="<f4").copy()
            per = len(cols) * NK
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
    with open(os.path.join(outdir, "index.json"), "w", newline="\n") as fh:
        json.dump(dict(mp_physics=52, n_cat=2, in3d=IN3D, in2d=IN2D,
                       out3d=OUT3D, out2d=OUT2D, cases=index), fh, indent=1)
    for row in index:
        print("%-26s ncol=%3d nk=%d nsteps=%2d dt=%4.0f %s"
              % (row["name"], row["ncol"], row["nk"], row["nsteps"],
                 row["dt"], row["sha256"][:16]))
    regs = {}
    for c in allc + ecols:
        regs[c.regime] = regs.get(c.regime, 0) + 1
    print("columns by regime:", regs, "total", sum(regs.values()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures-2cat")
