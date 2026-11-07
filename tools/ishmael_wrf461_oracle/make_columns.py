"""Column inputs for the WRF v4.6.1 ISHMAEL oracle (mp_physics = 55).

Writes one little-endian stream file the Fortran driver (run_ishmael.F90)
and the CUDA harness both read, so the two sides integrate the same bytes:

    int32 magic 'ISHM', ncol, nz
    float32 (nz, ncol) x 19   p, dz, th, qv, qc, qr, nr,
                              qi1, ni1, ai1, ci1, qi2, ni2, ai2, ci2,
                              qi3, ni3, ai3, ci3
    float32 (ncol,) x 2       rainnc, snownc   (accumulators before step 1)

(nz, ncol) C order is Fortran (ncol, nz) order, i fastest.  The column
families are listed in FAMILIES; each column is deterministic in its index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAGIC_IN = 1229146957
NZ = 48
INPUT_FIELDS = ("p", "dz", "th", "qv", "qc", "qr", "nr",
                "qi1", "ni1", "ai1", "ci1", "qi2", "ni2", "ai2", "ci2",
                "qi3", "ni3", "ai3", "ci3")
RD, CP = 287.15, 1004.0
RCP = 0.285856574
G = 9.8
AO = 0.1e-6

#: (first column, count, family) -- 96 columns in all.
FAMILIES = (
    (0, 16, "deep-convective"),
    (16, 16, "stratiform-melting"),
    (32, 12, "cirrus"),
    (44, 16, "mixed-phase-riming"),
    (60, 12, "cold-surface"),
    (72, 24, "edge-case"),
)


def family_of(col: int) -> str:
    for first, count, name in FAMILIES:
        if first <= col < first + count:
            return name
    raise ValueError(col)


def _qvs_liquid(t, p):
    # Bolton; only used to place humidity, never compared.
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def _qvs_ice(t, p):
    es = 611.2 * np.exp(21.8745584 * (t - 273.16) / (t - 7.66))
    return 0.622 * es / np.maximum(p - es, 1.0)


def _grid(rng, stretch):
    dz = np.minimum(70.0 * stretch * 1.075 ** np.arange(NZ), 650.0)
    zw = np.concatenate([[0.0], np.cumsum(dz)])
    zm = 0.5 * (zw[1:] + zw[:-1])
    return dz, zm


def _thermo(zm, tsfc, ztrop, psfc):
    t = np.where(zm < ztrop, tsfc - 6.5e-3 * zm,
                 tsfc - 6.5e-3 * ztrop + 1.0e-3 * (zm - ztrop))
    # hydrostatic in float64 on the mass levels
    p = np.empty_like(zm)
    p[0] = psfc * np.exp(-G * zm[0] / (RD * t[0]))
    for k in range(1, NZ):
        tbar = 0.5 * (t[k] + t[k - 1])
        p[k] = p[k - 1] * np.exp(-G * (zm[k] - zm[k - 1]) / (RD * tbar))
    return t, p


def _ice(qi, ni, an, delta):
    """ISHMAEL moments from mass, number, a-axis and delta*."""
    cn = AO ** (1.0 - delta) * an ** delta
    ai = an * an * cn * ni
    ci = cn * cn * an * ni
    return qi, ni, ai, ci


def _column(col: int):
    rng = np.random.default_rng(55_000 + col)
    fam = family_of(col)
    u = rng.uniform
    lu = lambda lo, hi, size=None: np.exp(rng.uniform(np.log(lo), np.log(hi), size))
    dz, zm = _grid(rng, u(0.85, 1.25))
    f = {name: np.zeros(NZ) for name in INPUT_FIELDS}
    f["dz"] = dz
    tc = None
    if fam == "deep-convective":
        t, p = _thermo(zm, u(298, 306), u(11e3, 15e3), u(99000, 101500))
        qv = _qvs_liquid(t, p) * np.clip(u(0.85, 1.03) - zm * u(0, 2e-5), 0.2, 1.05)
        warm = t > 273.15
        cloud = (zm > 800) & (zm < u(6e3, 11e3))
        f["qc"] = np.where(cloud, lu(1e-5, 2.5e-3, NZ), 0.0)
        f["qr"] = np.where(zm < u(4e3, 8e3), lu(1e-6, 4e-3, NZ), 0.0)
        f["nr"] = f["qr"] * lu(1e5, 1e8, NZ)
        cold = ~warm
        for n, (qlo, qhi) in zip((1, 2, 3), ((1e-6, 4e-4), (1e-6, 6e-4), (1e-6, 2e-3))):
            qi = np.where(cold & (u(0, 1, NZ) < 0.85), lu(qlo, qhi, NZ), 0.0)
            ni = qi * lu(1e5, 1e9, NZ)
            an = lu(5e-6, 6e-4, NZ)
            dl = u(0.6, 1.25, NZ) if n < 3 else u(0.6, 1.0, NZ)
            moments = _ice(qi, ni, an, dl)
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    elif fam == "stratiform-melting":
        t, p = _thermo(zm, u(277, 290), u(9e3, 12e3), u(98000, 102000))
        qv = np.minimum(_qvs_liquid(t, p), _qvs_ice(t, p) * 1.1) * u(0.88, 1.02, NZ)
        warm = t > 273.15
        f["qr"] = np.where(warm, lu(1e-6, 8e-4, NZ), 0.0)
        f["nr"] = f["qr"] * lu(1e4, 1e7, NZ)
        f["qc"] = np.where(u(0, 1, NZ) < 0.3, lu(1e-7, 3e-4, NZ), 0.0)
        near = (t > 268) & (t < 281)
        for n, (qlo, qhi) in zip((1, 2, 3), ((1e-7, 1e-4), (1e-7, 1e-4), (1e-6, 1.5e-3))):
            qi = np.where((~warm | near) & (zm < 9e3) & (u(0, 1, NZ) < 0.9),
                          lu(qlo, qhi, NZ), 0.0)
            ni = qi * lu(1e4, 1e8, NZ)
            an = lu(2e-5, 1.5e-3, NZ)
            moments = _ice(qi, ni, an, u(0.6, 1.2, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    elif fam == "cirrus":
        t, p = _thermo(zm, u(285, 300), u(10e3, 14e3), u(99000, 101500))
        rhi = u(0.7, 1.35, NZ)
        qv = np.where(t < 238, _qvs_ice(t, p) * rhi,
                      _qvs_liquid(t, p) * u(0.3, 0.8, NZ))
        high = t < u(225, 240)
        for n in (1, 2):
            qi = np.where(high & (u(0, 1, NZ) < 0.8), lu(1e-8, 5e-5, NZ), 0.0)
            ni = qi * lu(1e6, 1e10, NZ)
            an = lu(3e-6, 2e-4, NZ)
            moments = _ice(qi, ni, an, u(0.55, 1.3, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
        if col % 2:
            qi = np.where(high, lu(1e-8, 1e-5, NZ), 0.0)
            moments = _ice(qi, qi * lu(1e4, 1e7, NZ), lu(1e-4, 1e-3, NZ), u(0.6, 1.0, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}3"] = val
    elif fam == "mixed-phase-riming":
        t, p = _thermo(zm, u(272, 282), u(8e3, 11e3), u(97000, 101000))
        qv = _qvs_liquid(t, p) * u(0.95, 1.02, NZ)
        layer = (t > 248) & (t < 273.15)
        f["qc"] = np.where(layer, lu(1e-5, 1.5e-3, NZ), 0.0)
        f["qr"] = np.where(u(0, 1, NZ) < 0.6, lu(1e-7, 1.2e-3, NZ), 0.0)
        f["nr"] = f["qr"] * lu(1e3, 1e7, NZ)
        for n in (1, 2, 3):
            qi = np.where((t < 273.15) & (u(0, 1, NZ) < 0.8), lu(1e-6, 1e-3, NZ), 0.0)
            ni = qi * lu(1e4, 1e8, NZ)
            an = lu(1e-5, 1e-3, NZ)
            moments = _ice(qi, ni, an, u(0.55, 1.3, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    elif fam == "cold-surface":
        t, p = _thermo(zm, u(228, 252), u(7e3, 9e3), u(99000, 104000))
        qv = _qvs_liquid(t, p) * u(0.6, 1.05, NZ)
        f["qc"] = np.where(u(0, 1, NZ) < 0.5, lu(1e-7, 4e-4, NZ), 0.0)
        f["qr"] = np.where(u(0, 1, NZ) < 0.3, lu(1e-8, 2e-4, NZ), 0.0)
        f["nr"] = f["qr"] * lu(1e3, 1e8, NZ)
        for n in (1, 2, 3):
            qi = np.where(u(0, 1, NZ) < 0.7, lu(1e-9, 3e-4, NZ), 0.0)
            ni = qi * lu(1e4, 1e10, NZ)
            an = lu(3e-6, 4e-4, NZ)
            moments = _ice(qi, ni, an, u(0.55, 1.3, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    else:
        return _edge(col, rng, dz, zm, f)
    f["p"], f["qv"] = p, qv
    f["th"] = t * (100000.0 / p) ** RCP
    return f


def _edge(col, rng, dz, zm, f):
    u = rng.uniform
    lu = lambda lo, hi, size=None: np.exp(rng.uniform(np.log(lo), np.log(hi), size))
    case = col - 72
    t, p = _thermo(zm, u(285, 300), 12e3, 100000.0)
    qv = _qvs_liquid(t, p) * 0.7
    if case == 0:     # dry, no condensate anywhere: the domicro=false arm
        qv = _qvs_liquid(t, p) * 0.5
    elif case == 1:   # supersaturated w.r.t. liquid with no condensate
        qv = _qvs_liquid(t, p) * 1.01
    elif case == 2:   # masses at and around QSMALL; numbers around QNSMALL
        vals = np.array([1e-12, 1.0000001e-12, 9.999999e-13, 1.1e-12, 2e-12, 5e-13])
        for shift, name in enumerate(("qc", "qr", "qi1", "qi2", "qi3")):
            f[name] = np.resize(np.roll(vals, shift), NZ)
        for name in ("nr", "ni1", "ni2", "ni3"):
            f[name] = np.resize(np.array([1.25e-7, 1.2e-7, 1.3e-7, 0.0]), NZ)
        for name in ("ai1", "ci1", "ai2", "ci2", "ai3", "ci3"):
            f[name] = np.resize(np.array([1e-24, 1e-20, 1e-12, 0.0]), NZ)
    elif case == 3:   # subnormals and signed zeros in every hydrometeor field
        vals = np.array([1e-40, -1e-40, 0.0, -0.0, 1.4e-45, 1.1754942e-38])
        for i, name in enumerate(INPUT_FIELDS[4:]):
            f[name] = np.resize(np.roll(vals, i), NZ)
    elif case == 4:   # small negative advection undershoots beside real ice
        for name in ("qc", "qr", "nr", "qi1", "ni1", "ai1", "ci1"):
            f[name] = np.where(np.arange(NZ) % 2, -lu(1e-14, 1e-9, NZ),
                               lu(1e-7, 1e-4, NZ))
        moments = _ice(lu(1e-6, 1e-4, NZ), lu(1e3, 1e6, NZ), lu(1e-5, 3e-4, NZ),
                       u(0.6, 1.2, NZ))
        for name, val in zip(("qi", "ni", "ai", "ci"), moments):
            f[f"{name}2"] = val
    elif case == 5:   # volume moments below the 1e-12 shape floor
        for n in (1, 2, 3):
            f[f"qi{n}"] = lu(1e-8, 1e-4, NZ)
            f[f"ni{n}"] = lu(1e2, 1e6, NZ)
            f[f"ai{n}"] = lu(1e-20, 1e-12, NZ)
            f[f"ci{n}"] = lu(1e-20, 1e-12, NZ) * (np.arange(NZ) % 3 > 0)
        t = np.linspace(290, 215, NZ)
    elif case == 6:   # extreme aspect ratios on both sides of the delta* clamps
        for n in (1, 2, 3):
            dl = np.where(np.arange(NZ) % 2, 0.3, 1.8)
            moments = _ice(lu(1e-6, 5e-4, NZ), lu(1e3, 1e6, NZ),
                           lu(2e-5, 8e-4, NZ), dl)
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
        t = np.linspace(285, 210, NZ)
    elif case == 7:   # very large particles: var_check size cap, aggregate breakup
        for n in (1, 2, 3):
            moments = _ice(lu(1e-4, 5e-3, NZ), lu(1e-2, 10.0, NZ),
                           lu(1e-3, 8e-3, NZ), u(0.7, 1.2, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
        t = np.linspace(280, 220, NZ)
    elif case == 8:   # very high ice number: the 1000/L cap and nucleation limiter
        for n in (1, 2, 3):
            moments = _ice(lu(1e-6, 1e-4, NZ), lu(1e8, 1e11, NZ),
                           lu(2e-6, 2e-5, NZ), u(0.6, 1.2, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
        qv = _qvs_liquid(t, p) * 1.02
        t = np.linspace(272, 215, NZ)
    elif case == 9:   # rain slope clamps at both ends
        f["qr"] = lu(1e-6, 5e-3, NZ)
        f["nr"] = np.where(np.arange(NZ) % 2, f["qr"] * 1e1, f["qr"] * 1e12)
        f["qc"] = lu(1e-7, 1e-3, NZ)
    elif case == 10:  # temperatures exactly on branch boundaries (p = 1e5 Pa)
        temps = [273.15, 273.16, 270.16, 268.16, 265.16, 238.15, 269.15,
                 253.15, 272.15, 214.15, 213.15, 212.15, 273.14, 274.15,
                 253.1499, 238.1501, 269.1499, 272.1501]
        t = np.resize(np.array(temps), NZ)
        p = np.full(NZ, 100000.0)
        qv = _qvs_liquid(t, p) * np.resize(np.array([1.0, 0.98, 1.01]), NZ)
        f["qc"] = np.resize(np.array([1e-4, 0.0, 1e-6, 1e-7]), NZ)
        f["qr"] = np.resize(np.array([2e-4, 1e-8, 0.0]), NZ)
        f["nr"] = f["qr"] * 1e6
        for n in (1, 2, 3):
            moments = _ice(np.resize(np.array([2e-4, 1e-6, 0.0, 1e-8]), NZ),
                           np.full(NZ, 1e5), np.full(NZ, 1e-4), np.full(NZ, 0.9 + 0.1 * n))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    elif case == 11:  # ice-rain freezing (qr, qi > 1e-4) and ice-rain melting
        t = np.linspace(283, 248, NZ)
        f["qr"] = lu(1.1e-4, 3e-3, NZ)
        f["nr"] = f["qr"] * lu(1e4, 1e6, NZ)
        for n in (1, 2, 3):
            moments = _ice(lu(1.1e-4, 2e-3, NZ), lu(1e3, 1e5, NZ),
                           lu(1e-4, 1e-3, NZ), u(0.6, 1.2, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
        qv = _qvs_liquid(t, p) * 0.97
    elif case == 12:  # melting ice in sub- and supersaturated warm air
        t = np.linspace(281, 271, NZ)
        qv = _qvs_liquid(t, p) * np.resize(np.array([0.6, 1.01, 0.9]), NZ)
        for n in (1, 2, 3):
            moments = _ice(lu(1e-7, 1e-3, NZ), lu(1e2, 1e6, NZ),
                           lu(1e-5, 2e-3, NZ), u(0.55, 1.3, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    elif case == 13:  # wet growth: heavy riming near -3 C on dense particles
        t = np.linspace(274, 262, NZ)
        qv = _qvs_liquid(t, p) * 1.0
        f["qc"] = lu(1e-3, 4e-3, NZ)
        f["qr"] = lu(1e-4, 3e-3, NZ)
        f["nr"] = f["qr"] * 1e5
        for n in (1, 2, 3):
            qi = lu(1e-4, 4e-3, NZ)
            an = lu(3e-4, 2e-3, NZ)
            ni = qi / (900.0 * 4.18879 * an ** 3)
            moments = _ice(qi, ni, an, np.full(NZ, 1.0))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    elif case == 14:  # the columnar (igr > 1) nucleation regime, -4 to -9 C
        t = np.resize(np.linspace(269.5, 264.0, 12), NZ)
        qv = _qvs_liquid(t, p) * 1.005
        f["qc"] = lu(1e-5, 5e-4, NZ)
        for n in (1, 2):
            moments = _ice(lu(1e-7, 1e-5, NZ), lu(1e3, 1e6, NZ),
                           lu(5e-6, 5e-5, NZ), u(0.6, 1.3, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    elif case == 15:  # below -60 C and the IGR table's far end
        t = np.linspace(225, 190, NZ)
        qv = _qvs_ice(t, p) * u(0.9, 1.3, NZ)
        for n in (1, 2, 3):
            moments = _ice(lu(1e-9, 1e-5, NZ), lu(1e5, 1e9, NZ),
                           lu(2e-6, 1e-4, NZ), u(0.55, 1.3, NZ))
            for name, val in zip(("qi", "ni", "ai", "ci"), moments):
                f[f"{name}{n}"] = val
    elif case == 16:  # very high cloud water: autoconversion and depletion limits
        f["qc"] = lu(3e-3, 1.2e-2, NZ)
        qv = _qvs_liquid(t, p) * 1.0
        f["qr"] = lu(1e-8, 1e-5, NZ)
        f["nr"] = f["qr"] * 1e7
    elif case == 17:  # homogeneous freezing below -35 C with cloud and rain
        t = np.linspace(245, 225, NZ)
        f["qc"] = lu(1e-6, 1e-3, NZ)
        f["qr"] = lu(1e-7, 5e-4, NZ)
        f["nr"] = f["qr"] * lu(1e4, 1e7, NZ)
        qv = _qvs_liquid(t, p) * 1.0
    else:             # fuzz: log-uniform across every field's plausible span
        t = u(200, 305) - 0.0 * zm + u(-8, 8, NZ)
        p = np.sort(u(5000, 104000, NZ))[::-1]
        qv = _qvs_liquid(np.maximum(t, 190), p) * u(0.2, 1.08, NZ)
        on = lambda: u(0, 1, NZ) < 0.6
        f["qc"] = np.where(on(), lu(1e-12, 5e-3, NZ), 0.0)
        f["qr"] = np.where(on(), lu(1e-12, 5e-3, NZ), 0.0)
        f["nr"] = np.where(on(), lu(1e-7, 1e9, NZ), 0.0)
        for n in (1, 2, 3):
            f[f"qi{n}"] = np.where(on(), lu(1e-12, 5e-3, NZ), 0.0)
            f[f"ni{n}"] = np.where(on(), lu(1e-7, 1e10, NZ), 0.0)
            f[f"ai{n}"] = np.where(on(), lu(1e-24, 1e-3, NZ), 0.0)
            f[f"ci{n}"] = np.where(on(), lu(1e-24, 1e-3, NZ), 0.0)
    f["p"], f["qv"] = p, qv
    f["th"] = t * (100000.0 / p) ** RCP
    return f


def build(ncol: int = 96):
    cols = [_column(c) for c in range(ncol)]
    arrays = {name: np.stack([c[name] for c in cols], axis=1).astype(np.float32)
              for name in INPUT_FIELDS}
    rng = np.random.default_rng(5555)
    rainnc = rng.uniform(0, 50, ncol).astype(np.float32)
    snownc = (rainnc * rng.uniform(0, 1, ncol)).astype(np.float32)
    rainnc[:8] = 0.0
    snownc[:8] = 0.0
    return arrays, rainnc, snownc


def write(path: Path, arrays, rainnc, snownc):
    nz, ncol = arrays["p"].shape
    with open(path, "wb") as fh:
        fh.write(np.array([MAGIC_IN, ncol, nz], dtype="<i4").tobytes())
        for name in INPUT_FIELDS:
            fh.write(np.ascontiguousarray(arrays[name], dtype="<f4").tobytes())
        fh.write(rainnc.astype("<f4").tobytes())
        fh.write(snownc.astype("<f4").tobytes())


def read(path: Path):
    raw = Path(path).read_bytes()
    magic, ncol, nz = np.frombuffer(raw[:12], dtype="<i4")
    if magic != MAGIC_IN:
        raise ValueError(f"{path}: not an ISHMAEL oracle input file")
    off = 12
    arrays = {}
    for name in INPUT_FIELDS:
        n = int(ncol) * int(nz)
        arrays[name] = np.frombuffer(raw, dtype="<f4", count=n,
                                     offset=off).reshape(nz, ncol).copy()
        off += 4 * n
    rainnc = np.frombuffer(raw, dtype="<f4", count=ncol, offset=off).copy()
    off += 4 * ncol
    snownc = np.frombuffer(raw, dtype="<f4", count=ncol, offset=off).copy()
    return arrays, rainnc, snownc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--ncol", type=int, default=96)
    args = ap.parse_args()
    arrays, rainnc, snownc = build(args.ncol)
    write(args.out, arrays, rainnc, snownc)
    meta = {"ncol": args.ncol, "nz": NZ,
            "families": [{"first": a, "count": b, "name": c} for a, b, c in FAMILIES]}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
