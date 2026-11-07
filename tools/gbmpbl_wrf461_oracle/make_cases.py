"""Write the GBM PBL oracle's input columns (``gbm-cases.bin``).

Every number the oracle consumes is constructed here, in float32, from a
fixed seed; the Fortran driver (``run_gbmpbl.F90``) reads the file, calls
WRF's own ``gbmpbl`` on it for several steps and writes what WRF wrote.
Nothing here is an expected value.

Two column sets, because ``gbmpbl`` takes one ``kte`` per call:

* set 0: 40 levels, model top near 13 km, dt 60 s: the regime coverage
  (stratocumulus-topped marine, convective daytime land, stable night,
  neutral windy, decoupled cloud layer, very shallow PBL, ice cloud, and
  randomly layered columns that drive pblhgt's layer merging).
* set 1: 16 levels, model top near 2.6 km, dt 20 s: PBLs that reach the
  model top.

File layout (little-endian, Fortran stream order; 3-D fields are
(nz + 1, ncol) C-order, i.e. Fortran ``(i, k)`` with ``i`` fastest):

    int32 nset
    per set:
      int32 ncol, nz, nstep
      float32 dt
      3-D: u v th qv qc qi p pi dz8w z rthraten tke
      2-D: psfc znt ust zol hol pbl regime psim psih xland hfx qfx tsk
           gz1oz0 wspd br
      int32 kpbl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

F = np.float32
R_D = 287.0
CP = 7.0 * R_D / 2.0
RCP = R_D / CP
G = 9.81
EP1 = 461.6 / 287.0 - 1.0

FIELDS_3D = ("u", "v", "th", "qv", "qc", "qi", "p", "pi", "dz8w", "z",
             "rthraten", "tke")
FIELDS_2D = ("psfc", "znt", "ust", "zol", "hol", "pbl", "regime", "psim",
             "psih", "xland", "hfx", "qfx", "tsk", "gz1oz0", "wspd", "br")


def _grid(nz: int, dz0: float, growth: float, dzmax: float):
    dz = np.minimum(dz0 * growth ** np.arange(nz), dzmax)
    zf = np.concatenate(([0.0], np.cumsum(dz)))
    zh = 0.5 * (zf[:-1] + zf[1:])
    return dz, zf, zh


def _column(zh, dz, theta, qv, qc, qi, u, v, rth, tke, sfc, psfc=101500.0):
    """Hydrostatic pressure for one column, then pack float32 fields."""
    nz = zh.size
    p = np.empty(nz)
    # Integrate the hypsometric relation from the surface with the layer's
    # virtual temperature; float64 here, the oracle sees the float32 cast.
    pf = psfc
    for k in range(nz):
        exn = (pf / 1.0e5) ** RCP
        tv = theta[k] * exn * (1.0 + EP1 * qv[k] - qc[k] - qi[k])
        p[k] = pf * np.exp(-G * 0.5 * dz[k] / (R_D * tv))
        pf = pf * np.exp(-G * dz[k] / (R_D * tv))
    col = {
        "u": u, "v": v, "th": theta, "qv": qv, "qc": qc, "qi": qi, "p": p,
        "dz8w": dz, "z": np.concatenate(([0.0], np.cumsum(dz)))[:-1],
        "rthraten": rth, "tke": tke,
    }
    col = {k: np.asarray(a, dtype=F) for k, a in col.items()}
    col["pi"] = ((col["p"].astype(np.float64) / 1.0e5) ** RCP).astype(F)
    sfc = dict(sfc)
    sfc["psfc"] = psfc
    return col, {k: F(sfc[k]) for k in sfc}


def _surface(rng, *, land: bool, hfx: float, qfx: float, tsk: float,
             psim: float, wspd: float, z1: float):
    znt = 0.1 if land else 2.0e-4
    gz1oz0 = np.log(z1 / znt)
    zol = -psim * 0.1
    return {
        "znt": znt, "ust": 0.3, "zol": zol, "hol": zol * 10.0,
        "pbl": 500.0, "regime": 3.0 if hfx > 0 else 1.0, "psim": psim,
        "psih": psim * 1.2, "xland": 1.0 if land else 2.0, "hfx": hfx,
        "qfx": qfx, "tsk": tsk, "gz1oz0": gz1oz0, "wspd": wspd,
        "br": -0.05 if hfx > 0 else 0.1,
    }


def _tke(zh, zi, spun: bool):
    if not spun:
        return np.full(zh.size, 0.001)       # gbmpblinit's cold start
    return np.maximum(0.6 * np.exp(-zh / max(zi, 50.0)), 0.001)


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def build_set(rng, nz, dz0, growth, dzmax, cases):
    dz, zf, zh = _grid(nz, dz0, growth, dzmax)
    cols, sfcs, tags = [], [], []
    for tag, fn in cases:
        col, sfc = fn(rng, zh, dz)
        cols.append(col)
        sfcs.append(sfc)
        tags.append(tag)
    return dz, zh, cols, sfcs, tags


# ---------------------------------------------------------------- regimes --

def stratocumulus(zi, dth, qcmax, wind, spun, sst):
    def make(rng, zh, dz):
        th = np.where(zh < zi, 289.0, 289.0 + dth + 0.005 * (zh - zi))
        qv = np.where(zh < zi, 9.0e-3 - 1.0e-6 * zh, 4.0e-3 * np.exp(-(zh - zi) / 3000.0))
        base = zi - 350.0
        frac = np.clip((zh - base) / max(zi - base, 1.0), 0.0, 1.0)
        qc = np.where((zh > base) & (zh < zi), qcmax * frac, 0.0)
        rth = np.zeros_like(zh)
        top = np.argmax(zh > zi) - 1
        rth[top] = -2.0e-4
        rth[max(top - 1, 0)] = -4.0e-5
        rth[zh > zi] = -1.5e-5
        u = np.full_like(zh, wind) + 0.001 * zh
        v = np.full_like(zh, -0.3 * wind)
        sfc = _surface(rng, land=False, hfx=12.0, qfx=3.0e-5, tsk=sst,
                       psim=0.3, wspd=wind, z1=zh[0])
        return _column(zh, dz, th, qv, qc, np.zeros_like(zh), u, v, rth,
                       _tke(zh, zi, spun), sfc, psfc=101800.0)
    return make


def convective(hfx, zi, spun, seed_jitter):
    def make(rng, zh, dz):
        th0 = 300.0
        th = np.where(zh < zi, th0 + 1.5 * np.exp(-zh / 40.0),
                      th0 + 1.5 + 0.0045 * (zh - zi))
        th = th + seed_jitter * rng.standard_normal(zh.size) * (zh < zi)
        qv = np.where(zh < zi, 1.2e-2, 1.2e-2 * np.exp(-(zh - zi) / 1500.0))
        u = 3.0 + 2.0 * np.tanh(zh / 1000.0)
        v = 1.0 + 0.0005 * zh
        sfc = _surface(rng, land=True, hfx=hfx, qfx=hfx / 2.5e6 * 0.8,
                       tsk=th0 * (1.0) + 6.0 * hfx / 350.0 - 3.0,
                       psim=0.5 + hfx / 300.0, wspd=float(np.hypot(u[0], v[0])),
                       z1=zh[0])
        return _column(zh, dz, th, qv, np.zeros_like(zh), np.zeros_like(zh),
                       u, v, -1.0e-5 * np.ones_like(zh), _tke(zh, zi, spun),
                       sfc, psfc=97000.0)
    return make


def stable(hfx, dth, wind, spun):
    def make(rng, zh, dz):
        th = 285.0 + dth * (1.0 - np.exp(-zh / 200.0)) + 0.004 * zh
        qv = 6.0e-3 * np.exp(-zh / 2500.0)
        u = wind * np.log1p(zh / 0.1) / np.log1p(zh[-1] / 0.1) * 2.0
        v = 0.5 * wind * np.sin(zh / 800.0)
        sfc = _surface(rng, land=True, hfx=hfx, qfx=-1.0e-6, tsk=282.0 - dth * 0.3,
                       psim=-1.0 - dth / 5.0, wspd=max(wind, 0.5), z1=zh[0])
        return _column(zh, dz, th, qv, np.zeros_like(zh), np.zeros_like(zh),
                       u, v, -2.0e-5 * np.ones_like(zh), _tke(zh, 150.0, spun),
                       sfc, psfc=100500.0)
    return make


def neutral(wind, hfx, spun):
    def make(rng, zh, dz):
        th = np.where(zh < 1000.0, 290.0, 290.0 + 0.003 * (zh - 1000.0))
        qv = 5.0e-3 * np.exp(-zh / 3000.0)
        u = wind * (0.6 + 0.4 * np.tanh(zh / 300.0))
        v = 0.2 * wind * np.ones_like(zh)
        sfc = _surface(rng, land=bool(wind < 18), hfx=hfx, qfx=1.0e-5, tsk=290.0 * 0.995,
                       psim=0.0, wspd=float(np.hypot(u[0], v[0])), z1=zh[0])
        return _column(zh, dz, th, qv, np.zeros_like(zh), np.zeros_like(zh),
                       u, v, np.zeros_like(zh), _tke(zh, 1000.0, spun),
                       sfc, psfc=101000.0)
    return make


def decoupled(zml, zb, zt, qcmax, spun):
    def make(rng, zh, dz):
        th = np.select([zh < zml, zh < zb, zh < zt],
                       [291.0, 291.0 + 0.004 * (zh - zml),
                        291.0 + 0.004 * (zb - zml)],
                       291.0 + 0.004 * (zb - zml) + 6.0 + 0.005 * (zh - zt))
        qv = np.select([zh < zml, zh < zt], [10.0e-3, 8.5e-3],
                       3.0e-3 * np.exp(-(zh - zt) / 2500.0))
        qc = np.where((zh > zb) & (zh < zt), qcmax * (zh - zb) / (zt - zb), 0.0)
        rth = np.zeros_like(zh)
        top = np.argmax(zh > zt) - 1
        rth[top] = -1.5e-4
        u = np.full_like(zh, 7.0)
        v = np.full_like(zh, 2.0)
        sfc = _surface(rng, land=False, hfx=8.0, qfx=2.5e-5, tsk=291.5,
                       psim=0.2, wspd=float(np.hypot(7.0, 2.0)), z1=zh[0])
        return _column(zh, dz, th, qv, qc, np.zeros_like(zh), u, v, rth,
                       _tke(zh, zt, spun), sfc, psfc=101700.0)
    return make


def shallow(zi, dth, fog, spun):
    def make(rng, zh, dz):
        th = np.where(zh < zi, 280.0, 280.0 + dth + 0.003 * (zh - zi))
        qv = np.where(zh < zi, 6.5e-3, 3.0e-3)
        qc = np.where((zh < zi) & fog, 1.0e-4, 0.0)
        rth = np.where(zh < zi, -1.0e-4 * fog, 0.0)
        u = np.where(zh < zi, 3.0, 8.0)
        v = np.where(zh < zi, 0.5, -2.0)
        sfc = _surface(rng, land=True, hfx=15.0 * (1 - fog), qfx=2.0e-6,
                       tsk=279.0, psim=-0.2, wspd=3.05, z1=zh[0])
        return _column(zh, dz, th, qv, qc, np.zeros_like(zh), u, v, rth,
                       _tke(zh, zi, spun), sfc, psfc=100800.0)
    return make


def icy(qimax, zlo, zhi, unstable_aloft, spun):
    def make(rng, zh, dz):
        th = 295.0 + 0.004 * zh
        if unstable_aloft:
            band = (zh > zlo) & (zh < zhi)
            th = np.where(band, 295.0 + 0.004 * zlo - 0.0005 * (zh - zlo), th)
            th = np.where(zh >= zhi, 295.0 + 0.004 * zlo - 0.0005 * (zhi - zlo)
                          + 0.006 * (zh - zhi), th)
        qv = 8.0e-3 * np.exp(-zh / 2200.0)
        qi = np.where((zh > zlo) & (zh < zhi), qimax, 0.0)
        qc = np.where((zh > zlo - 500.0) & (zh < zlo), 5.0e-5, 0.0)
        rth = np.where((zh > zhi - 300.0) & (zh < zhi), -8.0e-5, -1.0e-5)
        u = 5.0 + 0.002 * zh
        v = -2.0 + 0.001 * zh
        sfc = _surface(rng, land=True, hfx=60.0, qfx=4.0e-5, tsk=297.0,
                       psim=0.6, wspd=float(np.hypot(u[0], v[0])), z1=zh[0])
        return _column(zh, dz, th, qv, qc, qi, u, v, rth, _tke(zh, 800.0, spun),
                       sfc, psfc=99000.0)
    return make


def layered(seed, depth, spun, land):
    def make(rng, zh, dz):
        r = np.random.default_rng(seed)
        th = np.empty(zh.size)
        th[0] = 293.0
        for k in range(1, zh.size):
            if zh[k] < depth:
                step = r.uniform(-0.6, 0.8)
            else:
                step = 0.006 * (zh[k] - zh[k - 1])
            th[k] = th[k - 1] + step
        qv = 1.0e-2 * np.exp(-zh / 2500.0) * (1.0 + 0.1 * r.standard_normal(zh.size))
        qv = np.maximum(qv, 1.0e-5)
        tk = th * (np.exp(-zh / 8000.0)) ** RCP
        pk = 1.0e5 * np.exp(-zh / 8000.0)
        qs = _qsat(tk, pk)
        qc = np.where(r.uniform(size=zh.size) < 0.25,
                      np.maximum(qv - 0.95 * qs, 0.0) + 2.0e-5 * (zh < depth), 0.0)
        rth = r.uniform(-1.5e-4, 2.0e-5, size=zh.size)
        u = 4.0 + 6.0 * r.standard_normal() + np.cumsum(r.uniform(-0.5, 0.7, zh.size))
        v = np.cumsum(r.uniform(-0.6, 0.6, zh.size))
        hfx = r.uniform(-40.0, 300.0)
        sfc = _surface(r, land=land, hfx=hfx, qfx=r.uniform(-1e-6, 1.2e-4),
                       tsk=th[0] + r.uniform(-4.0, 6.0) - 2.0,
                       psim=r.uniform(-2.0, 2.0), wspd=float(np.hypot(u[0], v[0])),
                       z1=zh[0])
        return _column(zh, dz, th, qv, qc, np.zeros_like(zh), u, v, rth,
                       _tke(zh, depth * 0.6, spun), sfc,
                       psfc=r.uniform(96000.0, 102000.0))
    return make


def calm(spun):
    def make(rng, zh, dz):
        th = 288.0 + 0.003 * zh
        qv = 5.0e-3 * np.exp(-zh / 2500.0)
        z = np.zeros_like(zh)
        sfc = _surface(rng, land=True, hfx=0.0, qfx=0.0, tsk=288.0,
                       psim=0.0, wspd=0.0, z1=zh[0])
        return _column(zh, dz, th, qv, z, z, z, z, z, _tke(zh, 300.0, spun),
                       sfc, psfc=101325.0)
    return make


def set0_cases():
    c = []
    for n, (zi, dth, qcm, w, sst) in enumerate(
            [(500, 6, 2e-4, 5, 288), (700, 10, 5e-4, 7, 290), (900, 8, 4e-4, 9, 291),
             (1100, 6, 3e-4, 6, 292), (1300, 12, 6e-4, 10, 289), (600, 4, 1e-4, 4, 287.5),
             (800, 9, 7e-4, 12, 290.5), (1000, 7, 2.5e-4, 3, 291.5),
             (1200, 5, 5e-4, 8, 290), (750, 11, 3.5e-4, 6, 289.5)]):
        c.append((f"sc-{n}", stratocumulus(zi, dth, qcm, w, n % 2 == 1, sst)))
    for n, (hfx, zi) in enumerate([(100, 800), (200, 1500), (350, 2500), (500, 1500),
                                   (150, 2500), (300, 800), (450, 2000), (250, 1200),
                                   (80, 600), (400, 3000), (220, 1800), (380, 1000)]):
        c.append((f"cbl-{n}", convective(hfx, zi, n % 2 == 0, 0.05 * (n % 3))))
    for n, (hfx, dth, w) in enumerate([(-10, 2, 1), (-30, 5, 3), (-60, 10, 8),
                                       (-20, 8, 1.5), (-45, 3, 5), (-5, 12, 0.6),
                                       (-35, 6, 10), (-15, 4, 2), (-50, 9, 4),
                                       (-25, 2.5, 6)]):
        c.append((f"stable-{n}", stable(hfx, dth, w, n % 3 != 0)))
    for n, (w, hfx) in enumerate([(10, 0), (15, 5), (20, -3), (25, 1),
                                  (12, -1), (18, 2), (22, 0), (30, 0.5)]):
        c.append((f"neutral-{n}", neutral(w, hfx, n % 2 == 0)))
    for n, (zml, zb, zt, qcm) in enumerate([(400, 1200, 1800, 3e-4), (300, 1000, 1500, 5e-4),
                                            (500, 1500, 2200, 2e-4), (450, 900, 1300, 6e-4),
                                            (350, 1300, 2000, 4e-4), (600, 1600, 2400, 3e-4),
                                            (250, 800, 1100, 2.5e-4), (400, 1100, 1700, 7e-4)]):
        c.append((f"decoupled-{n}", decoupled(zml, zb, zt, qcm, n % 2 == 1)))
    for n, (zi, dth, fog) in enumerate([(80, 8, 0), (120, 12, 1), (150, 15, 0), (200, 10, 1),
                                        (100, 6, 0), (180, 9, 0), (90, 14, 1), (130, 7, 0)]):
        c.append((f"shallow-{n}", shallow(zi, dth, bool(fog), n % 2 == 0)))
    for n, (qim, zlo, zhi, una) in enumerate([(5e-5, 8000, 10000, 0), (2e-4, 3000, 5000, 1),
                                              (1e-4, 6000, 7500, 1), (3e-5, 9000, 11000, 0),
                                              (1.5e-4, 2500, 4000, 1), (8e-5, 5000, 6500, 0)]):
        c.append((f"ice-{n}", icy(qim, zlo, zhi, bool(una), n % 2 == 0)))
    for n in range(10):
        c.append((f"layered-{n}", layered(1000 + n, 1500 + 300 * n, n % 2 == 0, n % 3 != 0)))
    c.append(("calm-0", calm(False)))
    c.append(("calm-1", calm(True)))
    return c


def set1_cases():
    c = []
    for n, (hfx, zi) in enumerate([(300, 3000), (450, 5000), (200, 2600)]):
        c.append((f"top-cbl-{n}", convective(hfx, zi, n % 2 == 0, 0.05)))
    for n, (zi, dth, qcm, w, sst) in enumerate([(2300, 8, 5e-4, 7, 290), (2450, 6, 3e-4, 9, 291)]):
        c.append((f"top-sc-{n}", stratocumulus(zi, dth, qcm, w, n == 0, sst)))
    c.append(("top-neutral-0", neutral(20, 0, True)))
    c.append(("top-neutral-1", neutral(12, 3, False)))
    for n in range(5):
        c.append((f"top-layered-{n}", layered(2000 + n, 5000, n % 2 == 1, n % 2 == 0)))
    return c


def write(path: Path, nstep0: int = 8, nstep1: int = 8) -> dict:
    rng = np.random.default_rng(20261008)
    sets = [
        (40, 30.0, 1.09, 800.0, 60.0, nstep0, set0_cases()),
        (16, 60.0, 1.06, 250.0, 20.0, nstep1, set1_cases()),
    ]
    meta = {"sets": []}
    with open(path, "wb") as fh:
        np.array([len(sets)], dtype="<i4").tofile(fh)
        for nz, dz0, growth, dzmax, dt, nstep, cases in sets:
            dz, zh, cols, sfcs, tags = build_set(rng, nz, dz0, growth, dzmax, cases)
            ncol = len(cols)
            np.array([ncol, nz, nstep], dtype="<i4").tofile(fh)
            np.array([dt], dtype="<f4").tofile(fh)
            for name in FIELDS_3D:
                arr = np.zeros((nz + 1, ncol), dtype="<f4")
                for i, col in enumerate(cols):
                    arr[:nz, i] = col[name]
                # level kme (nz+1) is never read by the scheme; give it the
                # top value so nothing in the file is a sentinel
                arr[nz] = arr[nz - 1]
                arr.tofile(fh)
            for name in FIELDS_2D:
                np.array([s[name] for s in sfcs], dtype="<f4").tofile(fh)
            np.zeros(ncol, dtype="<i4").tofile(fh)          # kpbl
            meta["sets"].append({"ncol": ncol, "nz": nz, "nstep": nstep,
                                 "dt": dt, "tags": tags})
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--meta", type=Path, required=True)
    args = ap.parse_args()
    meta = write(args.out)
    args.meta.write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"sets": [(s["ncol"], s["nz"]) for s in meta["sets"]],
                      "sha256": meta["sha256"]}))


if __name__ == "__main__":
    main()
