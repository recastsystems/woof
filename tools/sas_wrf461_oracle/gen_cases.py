"""Column cases for the WRF v4.6.1 SAS (cu_physics = 94) oracle.

Writes ``cases.bin``: the WRF-side single-precision inputs of ``CU_SAS``
(module_cu_sas.F:7) for every column, one record per column, little endian:

    int32  ncase
    per case:
        int32   nz, stepcu
        float32 dt, pgcon, xland
        float32 t3d[nz] qv[nz] qc[nz] qi[nz] u[nz] v[nz] w[nz+1]
                pcps[nz] pi[nz] rho[nz] dz8w[nz] p8w[nz+1]
        int32   category

The categories are the brief's list: deep, capped, shallow-only, dry, land
and ocean, two-level columns, long time steps, plus a high-top family whose
upper levels sit above 4 % of the surface pressure (so sascnvn's ``kmax`` is
below ``km``), cold columns (ice/water split of the detrained condensate) and
subsiding columns.  The soundings are synthetic but hydrostatic; the oracle
does not need them to be climatology, it needs every branch of the scheme
exercised.  Deterministic: one seed, numpy's PCG64.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

RD = 287.04
RV = 461.6
CP = 1004.5
G = 9.81

CATEGORIES = {
    0: "deep-ocean",
    1: "deep-land",
    2: "capped",
    3: "shallow-only",
    4: "dry",
    5: "two-level",
    6: "long-dt",
    7: "high-top",
    8: "cold",
    9: "subsiding",
}


def qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def eta_levels(nz, rng, stretch):
    s = np.linspace(0.0, 1.0, nz + 1)
    znw = 1.0 - s ** stretch
    znw[0], znw[-1] = 1.0, 0.0
    return znw


def column(rng, cat):
    nz = {5: 2, 7: 60}.get(cat, int(rng.choice([35, 45, 50])))
    p_top = 1000.0 if cat == 7 else 5000.0
    ps = float(rng.uniform(95000.0, 102000.0))
    znw = eta_levels(nz, rng, stretch=float(rng.uniform(1.3, 1.8)) if nz > 2 else 1.0)
    p8w = p_top + znw * (ps - p_top)
    pcps = 0.5 * (p8w[:-1] + p8w[1:])

    # temperature against pressure: surface value, lapse in the troposphere,
    # isothermal stratosphere, optional inversion
    tsfc = {8: rng.uniform(245.0, 262.0), 4: rng.uniform(285.0, 305.0)}.get(
        cat, rng.uniform(292.0, 303.0))
    gamma = rng.uniform(5.8, 7.2) / 1000.0     # K/m
    trop_p = rng.uniform(12000.0, 20000.0)
    # height from pressure with a constant-lapse atmosphere
    z = (tsfc / gamma) * (1.0 - (pcps / ps) ** (RD * gamma / G))
    t = tsfc - gamma * z
    ttrop = tsfc - gamma * (tsfc / gamma) * (1.0 - (trop_p / ps) ** (RD * gamma / G))
    t = np.where(pcps < trop_p, ttrop + 2.0e-4 * (trop_p - pcps) / 10.0, t)
    if cat == 2:
        pinv = rng.uniform(80000.0, 88000.0)
        t = t + np.where(pcps < pinv, rng.uniform(6.0, 10.0) * np.exp(-(pinv - pcps) / 15000.0), 0.0)
    if cat == 3:
        pinv = rng.uniform(72000.0, 80000.0)
        t = t + np.where(pcps < pinv, rng.uniform(2.0, 4.0) * np.exp(-(pinv - pcps) / 12000.0), 0.0)
    t = t + rng.normal(0.0, 0.3, size=nz)

    # relative humidity profile
    if cat == 4:
        rh = rng.uniform(0.05, 0.25) * np.ones(nz)
    elif cat == 3:
        rh = np.where(pcps > ps - rng.uniform(12000.0, 18000.0),
                      rng.uniform(0.88, 0.98), rng.uniform(0.15, 0.3))
    elif cat == 2:
        rh = np.where(pcps > 85000.0, rng.uniform(0.8, 0.95), rng.uniform(0.2, 0.5))
    else:
        rh_low = rng.uniform(0.75, 0.97)
        rh_mid = rng.uniform(0.4, 0.9)
        frac = (ps - pcps) / (ps - p_top)
        rh = rh_low + (rh_mid - rh_low) * np.clip(frac / 0.45, 0.0, 1.0)
        rh = np.where(pcps < trop_p, 0.05, rh)
    qv = np.clip(rh * qsat(t, pcps), 1.0e-7, 0.03)

    tv = t * (1.0 + 0.608 * qv)
    dz8w = RD * tv / G * np.log(p8w[:-1] / p8w[1:])
    rho = pcps / (RD * tv)
    pi = (pcps / 1.0e5) ** (RD / CP)

    # condensate in a few layers
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    if rng.uniform() < 0.6:
        lay = (pcps > 50000.0) & (pcps < 85000.0)
        qc = np.where(lay, rng.uniform(0.0, 3.0e-4, size=nz), 0.0)
        cold = t < 260.0
        qi = np.where(cold & (pcps > 15000.0), rng.uniform(0.0, 1.0e-4, size=nz), 0.0)

    # vertical velocity at interfaces: w(1) = 0
    zf = np.concatenate([[0.0], np.cumsum(dz8w)])
    amp = {9: -rng.uniform(0.05, 0.6), 4: rng.uniform(-0.1, 0.1)}.get(
        cat, rng.uniform(-0.05, 1.5))
    depth = max(zf[-1], 1.0)
    w = amp * np.sin(np.pi * np.clip(zf / min(depth, 14000.0), 0.0, 1.0))
    w[0] = 0.0
    # winds with shear
    u0 = rng.uniform(-10.0, 10.0)
    v0 = rng.uniform(-10.0, 10.0)
    shear = rng.uniform(0.0, 4.0e-3)
    zc = 0.5 * (zf[:-1] + zf[1:])
    u = u0 + shear * zc * np.cos(rng.uniform(0, 2 * np.pi))
    v = v0 + shear * zc * np.sin(rng.uniform(0, 2 * np.pi))

    xland = 1.0 if cat == 1 else 2.0 if cat == 0 else float(rng.choice([1.0, 2.0]))
    if cat == 6:
        dt, stepcu = float(rng.choice([180.0, 300.0])), int(rng.choice([5, 6, 10]))
    else:
        dt = float(rng.choice([18.0, 30.0, 60.0, 90.0]))
        stepcu = int(rng.choice([1, 1, 3, 5]))
    pgcon = float(rng.choice([0.55, 0.55, 0.2, 0.7, 0.0]))
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return dict(nz=nz, stepcu=stepcu, dt=dt, pgcon=pgcon, xland=xland,
                t3d=f32(t), qv=f32(qv), qc=f32(qc), qi=f32(qi), u=f32(u),
                v=f32(v), w=f32(w), pcps=f32(pcps), pi=f32(pi), rho=f32(rho),
                dz8w=f32(dz8w), p8w=f32(p8w), category=cat)


ORDER = ("t3d", "qv", "qc", "qi", "u", "v", "w", "pcps", "pi", "rho",
         "dz8w", "p8w")


def write_cases(cases, path: Path) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<i", len(cases)))
        for c in cases:
            fh.write(struct.pack("<ii", c["nz"], c["stepcu"]))
            fh.write(struct.pack("<fff", c["dt"], c["pgcon"], c["xland"]))
            for name in ORDER:
                fh.write(np.ascontiguousarray(c[name], dtype="<f4").tobytes())
            fh.write(struct.pack("<i", c["category"]))


def read_cases(path: Path):
    data = Path(path).read_bytes()
    off = 0

    def take(fmt):
        nonlocal off
        vals = struct.unpack_from(fmt, data, off)
        off += struct.calcsize(fmt)
        return vals

    (n,) = take("<i")
    cases = []
    for _ in range(n):
        nz, stepcu = take("<ii")
        dt, pgcon, xland = take("<fff")
        c = dict(nz=nz, stepcu=stepcu, dt=dt, pgcon=pgcon, xland=xland)
        for name in ORDER:
            m = nz + 1 if name in ("w", "p8w") else nz
            c[name] = np.frombuffer(data, dtype="<f4", count=m, offset=off).copy()
            off += 4 * m
        (c["category"],) = take("<i")
        cases.append(c)
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-category", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    cases = [column(rng, cat) for cat in CATEGORIES
             for _ in range(args.per_category)]
    write_cases(cases, args.out)
    print(f"wrote {len(cases)} columns to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
