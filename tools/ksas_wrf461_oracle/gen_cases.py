"""Column fixtures for the KIAPS SAS (cu_physics = 14) WRF v4.6.1 oracle.

Writes ``cases.bin``: every input ``cu_ksas`` reads, for a set of CALL
GROUPS.  A group is one ``cu_ksas`` call (one ``j`` row): ``dx``, ``dt``,
``stepcu`` and ``mp_physics`` are per-call scalars in WRF, so each axis value
is its own group.  The same bytes feed ``run_ksas.F90`` (the real WRF object)
and the GPU test, so the two sides cannot disagree about an input.

Columns are synthetic soundings built to reach every exit of ``nsas2d``:
deep tropical and mid-latitude severe columns that rain, capped columns whose
cloud is shallower than 150 hPa (the depth refusal), dry and stable columns
that never find a level of free convection, strong-shear columns that drive
``vshear`` past the polynomial's range, subsiding columns that the ``xmb <
pdot/g`` test turns off, land (XLAND 1) and water (XLAND 2), coarse and
gray-zone spacing.  The category of every column is recorded in
``cases.json`` so a report can count outcomes per category.

Binary layout (little-endian, stream):
  int32 ngroups
  per group:
    int32 ncol, nz, stepcu, mp_physics, f_qc, f_qi, dx_factor_nsas, itimestep
    float32 dt, dx
    float32 znu[nz]
    float32 [nz, ncol] C-order: p3d pi3d t3d qv3d qc3d qi3d rho3d dz8w u3d v3d
    float32 [nz+1, ncol] C-order: p3di w
    float32 [ncol]: xland hpbl hpbl_hold hfx qfx
A C-order [nz, ncol] block is Fortran A(ncol, nz), which is how the driver
reads it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

F32 = np.float32
RD, RV, CP, G = 287.0, 461.6, 1004.5, 9.81

LEVEL_FIELDS = ("p3d", "pi3d", "t3d", "qv3d", "qc3d", "qi3d", "rho3d",
                "dz8w", "u3d", "v3d")
IFACE_FIELDS = ("p3di", "w")
SFC_FIELDS = ("xland", "hpbl", "hpbl_hold", "hfx", "qfx")

#: (dx m, dt s, stepcu, mp_physics, nz).  mp 1 and 3 give ncloud = 1, mp 0
#: gives ncloud = 0, anything else ncloud = 2 (module_cu_ksas.F:145-151).
GROUPS = (
    (30000.0, 150.0, 1, 8, 45),
    (15000.0, 60.0, 5, 8, 45),      # stepcu > 1: delt = dt*stepcu
    (12000.0, 60.0, 1, 6, 50),
    (9000.0, 45.0, 1, 1, 45),       # Kessler: ncloud 1, no QI
    (5000.0, 30.0, 1, 8, 50),       # the 5 km boundary of the sigma branch
    (4000.0, 24.0, 1, 3, 45),       # WSM3: ncloud 1
    (3000.0, 18.0, 1, 8, 60),
    (1000.0, 6.0, 1, 8, 60),
    (500.0, 3.0, 1, 10, 45),
    (250.0, 1.5, 1, 0, 45),         # mp 0: ncloud 0
    (27000.0, 120.0, 3, 8, 35),
    (2000.0, 12.0, 1, 8, 50),
)

CATEGORIES = ("deep-tropical", "severe-midlat", "capped-shallow",
              "dry-stable", "high-shear", "subsiding", "moist-neutral",
              "random")


def eta_levels(nz: int) -> np.ndarray:
    """A WRF-like stretched full-level eta set, znw[0] = 1, znw[nz] = 0."""
    x = np.linspace(0.0, 1.0, nz + 1)
    znw = 1.0 - (0.35 * x + 0.65 * x ** 1.6)
    znw[0], znw[-1] = 1.0, 0.0
    return znw.astype(np.float64)


def qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


def column(rng, cat: str, nz: int, znw: np.ndarray):
    """One sounding.  Returns per-level and per-interface arrays (float64)."""
    ptop = 5000.0
    psfc = rng.uniform(92000.0, 101500.0)
    znu = 0.5 * (znw[:-1] + znw[1:])
    p8w = znw * (psfc - ptop) + ptop
    p = znu * (psfc - ptop) + ptop

    t0 = {"deep-tropical": rng.uniform(298, 304),
          "severe-midlat": rng.uniform(294, 303),
          "capped-shallow": rng.uniform(290, 300),
          "dry-stable": rng.uniform(270, 290),
          "high-shear": rng.uniform(292, 302),
          "subsiding": rng.uniform(292, 302),
          "moist-neutral": rng.uniform(288, 298),
          "random": rng.uniform(275, 304)}[cat]
    zml = rng.uniform(300.0, 2200.0)
    lapse_ml = rng.uniform(8.0, 9.8) / 1000.0
    lapse_ft = {"dry-stable": rng.uniform(3.0, 5.0),
                "moist-neutral": rng.uniform(5.5, 6.3)}.get(
                    cat, rng.uniform(6.0, 7.8)) / 1000.0
    ztp = rng.uniform(11000.0, 16500.0)
    cap = (rng.uniform(2.0, 6.0) if cat == "capped-shallow"
           else rng.uniform(0.0, 1.2) if cat in ("severe-midlat", "random")
           else 0.0)
    rh_ml = {"deep-tropical": rng.uniform(0.80, 0.96),
             "severe-midlat": rng.uniform(0.70, 0.92),
             "capped-shallow": rng.uniform(0.80, 0.98),
             "dry-stable": rng.uniform(0.15, 0.45),
             "high-shear": rng.uniform(0.70, 0.92),
             "subsiding": rng.uniform(0.70, 0.92),
             "moist-neutral": rng.uniform(0.85, 0.99),
             "random": rng.uniform(0.3, 0.97)}[cat]
    rh_ft = {"deep-tropical": rng.uniform(0.55, 0.85),
             "severe-midlat": rng.uniform(0.25, 0.6),
             "capped-shallow": rng.uniform(0.1, 0.4),
             "dry-stable": rng.uniform(0.05, 0.3),
             "moist-neutral": rng.uniform(0.7, 0.95)}.get(
                 cat, rng.uniform(0.2, 0.8))

    def tprof(z):
        if z <= zml:
            return t0 - lapse_ml * z
        tml = t0 - lapse_ml * zml + (cap if z > zml else 0.0)
        if z <= ztp:
            return tml - lapse_ft * (z - zml)
        return tml - lapse_ft * (ztp - zml) + 0.0015 * (z - ztp)

    # Hydrostatic heights at interfaces from the mid-level temperatures.
    z8w = np.zeros(nz + 1)
    t = np.zeros(nz)
    for k in range(nz):
        tk = tprof(z8w[k])
        for _ in range(3):
            dz = RD * tk / G * np.log(p8w[k] / p8w[k + 1])
            tk = tprof(z8w[k] + 0.5 * dz)
        t[k] = tk
        z8w[k + 1] = z8w[k] + RD * tk / G * np.log(p8w[k] / p8w[k + 1])
    zmid = 0.5 * (z8w[:-1] + z8w[1:])
    rh = np.where(zmid <= zml, rh_ml,
                  rh_ft * np.exp(-np.maximum(zmid - zml, 0.0) / 9000.0))
    if cat in ("severe-midlat", "random") and rng.random() < 0.5:
        lo, hi = zml + 500.0, zml + 500.0 + rng.uniform(800.0, 2500.0)
        rh = np.where((zmid > lo) & (zmid < hi), rh * 0.3, rh)
    rh = np.clip(rh + rng.normal(0.0, 0.02, nz), 0.02, 1.02)
    qv = rh * qsat(t, p)
    qv = np.where(zmid > ztp, np.minimum(qv, 5.0e-6), qv)
    tv = t * (1.0 + 0.608 * qv)
    rho = p / (RD * tv)
    pi = (p / 1.0e5) ** (RD / CP)

    wmax = {"deep-tropical": rng.uniform(0.0, 1.5),
            "severe-midlat": rng.uniform(0.2, 2.5),
            "subsiding": rng.uniform(-1.5, -0.3),
            "dry-stable": rng.uniform(-0.3, 0.1)}.get(
                cat, rng.uniform(-0.5, 1.0))
    htop = rng.uniform(6000.0, 14000.0)
    w = wmax * np.sin(np.pi * np.clip(z8w / htop, 0.0, 1.0))
    w = w + rng.normal(0.0, 0.05, nz + 1)
    w[0] = 0.0

    shear = (rng.uniform(25.0, 60.0) if cat == "high-shear"
             else rng.uniform(0.0, 20.0))
    u0, v0 = rng.uniform(-10, 10), rng.uniform(-10, 10)
    turn = rng.uniform(0.0, np.pi / 2)
    s = np.clip(zmid / 6000.0, 0.0, 1.6)
    u = u0 + shear * s * np.cos(turn * s)
    v = v0 + shear * s * np.sin(turn * s)
    if cat == "high-shear":
        u = u + rng.normal(0.0, 3.0, nz)

    qc = np.zeros(nz)
    qi = np.zeros(nz)
    if rng.random() < 0.4:
        band = (zmid > zml) & (zmid < zml + rng.uniform(1000, 5000))
        qc = np.where(band & (t > 253.0), rng.uniform(1e-5, 4e-4), 0.0)
        qi = np.where((zmid > 5000.0) & (t < 263.0) & (rng.random() < 0.7),
                      rng.uniform(1e-6, 1e-4), 0.0)

    hpbl = min(zml * rng.uniform(0.8, 1.2), 3500.0)
    hpbl_hold = max(hpbl - rng.uniform(-200.0, 400.0), 20.0)
    return {
        "p3d": p, "pi3d": pi, "t3d": t, "qv3d": qv, "qc3d": qc, "qi3d": qi,
        "rho3d": rho, "dz8w": np.diff(z8w), "u3d": u, "v3d": v,
        "p3di": p8w, "w": w,
        "hpbl": hpbl, "hpbl_hold": hpbl_hold,
        "hfx": rng.uniform(-20, 450), "qfx": rng.uniform(0, 3e-4),
    }


def build(seed: int, ncol: int):
    rng = np.random.default_rng(seed)
    groups = []
    for gi, (dx, dt, stepcu, mp, nz) in enumerate(GROUPS):
        znw = eta_levels(nz)
        znu = (0.5 * (znw[:-1] + znw[1:])).astype(F32)
        cols, cats = [], []
        for ic in range(ncol):
            cat = CATEGORIES[ic % len(CATEGORIES)]
            cols.append(column(rng, cat, nz, znw))
            cats.append(cat)
        arrays = {}
        for name in LEVEL_FIELDS + IFACE_FIELDS:
            arrays[name] = np.ascontiguousarray(
                np.stack([c[name] for c in cols], axis=1).astype(F32))
        arrays["xland"] = np.array([1.0 if (ic // 8) % 2 == 0 else 2.0
                                    for ic in range(ncol)], dtype=F32)
        for name in ("hpbl", "hpbl_hold", "hfx", "qfx"):
            arrays[name] = np.array([c[name] for c in cols], dtype=F32)
        groups.append({
            "ncol": ncol, "nz": nz, "stepcu": stepcu, "mp_physics": mp,
            "f_qc": 1 if mp != 0 else 0,
            "f_qi": 0 if mp in (0, 1, 3) else 1,
            "dx_factor_nsas": gi % 2, "itimestep": 1 + gi,
            "dt": dt, "dx": dx, "znu": znu, "arrays": arrays,
            "categories": cats,
        })
    return groups


def write(groups, path: Path) -> None:
    with open(path, "wb") as fh:
        np.array([len(groups)], dtype="<i4").tofile(fh)
        for g in groups:
            np.array([g["ncol"], g["nz"], g["stepcu"], g["mp_physics"],
                      g["f_qc"], g["f_qi"], g["dx_factor_nsas"],
                      g["itimestep"]], dtype="<i4").tofile(fh)
            np.array([g["dt"], g["dx"]], dtype="<f4").tofile(fh)
            g["znu"].astype("<f4").tofile(fh)
            for name in LEVEL_FIELDS + IFACE_FIELDS + SFC_FIELDS:
                g["arrays"][name].astype("<f4").tofile(fh)


def read(path: Path):
    """Inverse of :func:`write`; the GPU test reads the oracle's bytes."""
    raw = Path(path).read_bytes()
    off = 0

    def take(dtype, count):
        nonlocal off
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=off)
        off += a.nbytes
        return a

    groups = []
    for _ in range(int(take("<i4", 1)[0])):
        ncol, nz, stepcu, mp, fqc, fqi, dxf, its = (int(v) for v in take("<i4", 8))
        dt, dx = (float(v) for v in take("<f4", 2))
        g = {"ncol": ncol, "nz": nz, "stepcu": stepcu, "mp_physics": mp,
             "f_qc": fqc, "f_qi": fqi, "dx_factor_nsas": dxf,
             "itimestep": its, "dt": np.float32(dt), "dx": np.float32(dx),
             "znu": take("<f4", nz).copy(), "arrays": {}}
        for name in LEVEL_FIELDS:
            g["arrays"][name] = take("<f4", nz * ncol).reshape(nz, ncol).copy()
        for name in IFACE_FIELDS:
            g["arrays"][name] = take("<f4", (nz + 1) * ncol).reshape(
                nz + 1, ncol).copy()
        for name in SFC_FIELDS:
            g["arrays"][name] = take("<f4", ncol).copy()
        groups.append(g)
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return groups


OUT_LEVEL = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten",
             "rvcuten")
OUT_SFC = ("raincv", "pratec", "hbot", "htop")


def read_outputs(path: Path, groups):
    """The oracle's answers, per group, in the order run_ksas.F90 writes."""
    raw = Path(path).read_bytes()
    off = 0
    result = []
    for g in groups:
        nz, ncol = g["nz"], g["ncol"]
        out = {}
        for name in OUT_LEVEL:
            n = nz * ncol
            out[name] = np.frombuffer(raw, "<f4", n, off).reshape(nz, ncol).copy()
            off += 4 * n
        for name in OUT_SFC:
            out[name] = np.frombuffer(raw, "<f4", ncol, off).copy()
            off += 4 * ncol
        result.append(out)
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--ncol", type=int, default=64)
    args = ap.parse_args()
    groups = build(args.seed, args.ncol)
    args.out.mkdir(parents=True, exist_ok=True)
    write(groups, args.out / "cases.bin")
    meta = [{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
             for k, v in g.items() if k not in ("arrays", "znu")}
            for g in groups]
    (args.out / "cases.json").write_text(
        json.dumps({"seed": args.seed, "groups": meta}, indent=1) + "\n",
        encoding="utf-8")
    print(f"wrote {len(groups)} groups x {args.ncol} columns to {args.out}")


if __name__ == "__main__":
    main()
