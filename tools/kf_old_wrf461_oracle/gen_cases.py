"""Column cases for the old Kain-Fritsch (cu_physics = 99) WRF oracle.

Writes the binary case file ``run_kf_old`` reads (layout below) and a JSON
sidecar naming each case's regime, so a reader of the oracle's results can
see which branch of KFPARA each column was built to reach.

The cases are synthetic hydrostatic soundings plus the two real HRRR
columns the cu_physics = 1 tests already carry
(tests/data/kf_real74_12z_columns.npz), each crossed with the driver knobs
KFCPS reads: DT, DX, STEPCU, CUDT, the adaptive-step flag, the four
species flags (warm_rain, F_QI, F_QS, F_QR) and the NCA hold.  Every array
is float32 at the point it is written, so the oracle and the GPU kernel
read the same bits.

Case file layout (little-endian, stream):
  int32 magic 0x4B464F4C, int32 version 1, int32 ncase
  per case:
    int32 nz, stepcu, adapt, warm_rain, f_qi, f_qs, f_qr, kf_edrates
    float32 dt, dx, cudt, nca
    float32 u[nz], v[nz], w[nz+1], t[nz], th[nz], qv[nz], dz8w[nz],
            p[nz], rho[nz], pi[nz], w0avg[nz]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x4B464F4C
VERSION = 1
FIELDS = ("u", "v", "w", "t", "th", "qv", "dz8w", "p", "rho", "pi",
          "w0avg")

RD = 287.0
RV = 461.6
CP = 7.0 * RD / 2.0
G = 9.81

#: (warm_rain, f_qi, f_qs, f_qr) for the WRF moist packages the species
#: branches of KFCPS/KFPARA distinguish.  Named by the mp_physics value
#: whose Registry package gives that set.
SPECIES_SETS = {
    "mp0": (0, 0, 0, 0),    # passiveqv: moist:qv
    "mp1": (1, 0, 0, 1),    # kesslerscheme: qv,qc,qr; warm_rain
    "mp3": (0, 0, 0, 1),    # wsm3scheme: qv,qc,qr (ice in qc/qr)
    "mp6": (0, 1, 1, 1),    # wsm6scheme: qv,qc,qr,qi,qs,qg
    "mp50": (0, 1, 0, 1),   # p3_1category: qv,qc,qr,qi
}


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    es = np.minimum(es, 0.5 * p)
    return 0.622 * es / (p - es)


def synthetic(rng, *, nz, regime):
    """One hydrostatic column in a named regime (float64 until written)."""
    ztop = rng.uniform(17500.0, 21000.0)
    z_ifc = ztop * np.linspace(0.0, 1.0, nz + 1) ** rng.uniform(1.15, 1.45)
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)
    psfc = rng.uniform(97000.0, 101800.0)
    if regime == "high-terrain":
        psfc = rng.uniform(80000.0, 86000.0)
    params = {
        "deep-moist": dict(ts=(298, 305), gbl=(8.0, 9.8), g=(5.6, 6.6),
                           ztrop=(14000, 16500), rhbl=(0.82, 0.96),
                           rhmid=(0.55, 0.85), cap=(0.0, 0.5)),
        "deep-cold-top": dict(ts=(300, 306), gbl=(8.5, 9.8), g=(6.2, 7.2),
                              ztrop=(15500, 17500), rhbl=(0.85, 0.97),
                              rhmid=(0.5, 0.8), cap=(0.0, 0.3)),
        "midlat": dict(ts=(290, 300), gbl=(8.5, 9.8), g=(6.8, 8.0),
                       ztrop=(10500, 13000), rhbl=(0.7, 0.92),
                       rhmid=(0.2, 0.55), cap=(0.0, 1.5)),
        "high-terrain": dict(ts=(286, 296), gbl=(9.0, 9.8), g=(7.0, 8.2),
                             ztrop=(11000, 13500), rhbl=(0.6, 0.9),
                             rhmid=(0.25, 0.6), cap=(0.0, 1.0)),
        "shallow": dict(ts=(295, 302), gbl=(8.5, 9.8), g=(5.5, 6.5),
                        ztrop=(13000, 15000), rhbl=(0.85, 0.96),
                        rhmid=(0.1, 0.3), cap=(5.0, 9.0)),
        "stable": dict(ts=(275, 290), gbl=(3.0, 5.5), g=(3.5, 5.0),
                       ztrop=(9000, 11000), rhbl=(0.4, 0.75),
                       rhmid=(0.2, 0.5), cap=(1.0, 4.0)),
        "dry": dict(ts=(295, 310), gbl=(9.0, 9.8), g=(7.5, 9.0),
                    ztrop=(11000, 14000), rhbl=(0.05, 0.25),
                    rhmid=(0.03, 0.15), cap=(0.0, 0.5)),
        "cold-season": dict(ts=(262, 274), gbl=(6.0, 9.0), g=(6.0, 8.0),
                            ztrop=(8000, 10000), rhbl=(0.8, 0.98),
                            rhmid=(0.6, 0.9), cap=(0.0, 0.5)),
    }[regime]
    ts = rng.uniform(*params["ts"])
    gbl = rng.uniform(*params["gbl"]) * 1.0e-3
    gamma = rng.uniform(*params["g"]) * 1.0e-3
    ztrop = rng.uniform(*params["ztrop"])
    zbl = rng.uniform(600.0, 1600.0)
    t = np.where(z < zbl, ts - gbl * z, ts - gbl * zbl - gamma * (z - zbl))
    ttrop = ts - gbl * zbl - gamma * (ztrop - zbl)
    t = np.where(z > ztrop, ttrop + 1.0e-3 * (z - ztrop), t)
    cap = rng.uniform(*params["cap"])
    zcap = rng.uniform(2200.0, 3200.0)
    t = t + cap * np.exp(-((z - zcap) / 450.0) ** 2)
    rhbl = rng.uniform(*params["rhbl"])
    rhmid = rng.uniform(*params["rhmid"])
    rh = np.where(z < zbl, rhbl,
                  rhmid + (rhbl - rhmid) * np.exp(-(z - zbl) / 1500.0))
    rh = np.where(z > 9000.0, np.minimum(rh, 0.35), rh)
    if regime == "shallow":
        rh = np.where((z > zcap - 400.0) & (z < zcap + 3000.0),
                      np.minimum(rh, 0.12), rh)
    p = np.empty(nz)
    pk = psfc
    qv = np.zeros(nz)
    for _ in range(3):
        pk = psfc
        for k in range(nz):
            tv = t[k] * (1.0 + 0.608 * qv[k])
            half = dz[k] * 0.5
            pk_mid = pk * np.exp(-G * half / (RD * tv))
            p[k] = pk_mid
            pk = pk_mid * np.exp(-G * half / (RD * tv))
        qv = np.maximum(rh * _qsat(t, p), 1.0e-7)
    tv = t * (1.0 + 0.608 * qv)
    rho = p / (RD * tv) * (1.0 + qv)
    pi = (p / 1.0e5) ** (RD / CP)
    th = t / pi
    u0 = rng.uniform(-8.0, 12.0)
    v0 = rng.uniform(-6.0, 8.0)
    shear = rng.uniform(0.0, 3.5e-3)
    u = u0 + shear * z * np.cos(0.0002 * z)
    v = v0 + 0.6 * shear * z
    wlobe = rng.uniform(-0.05, 0.35)
    zw = rng.uniform(800.0, 3000.0)
    w = (wlobe * np.exp(-((z_ifc - zw) / 1500.0) ** 2)
         + rng.normal(0.0, 0.02, nz + 1))
    w[0] = 0.0
    w0avg = (0.5 * (w[:-1] + w[1:]) * rng.uniform(0.3, 1.2)
             + rng.normal(0.0, 0.005, nz))
    return dict(u=u, v=v, w=w, t=t, th=th, qv=qv, dz8w=dz, p=p, rho=rho,
                pi=pi, w0avg=w0avg)


def wide(rng, *, nz):
    """A column with every sounding knob drawn independently and widely.

    The curated regimes above are each built to reach one family of
    branches; these exist to reach the branches no named regime reliably
    does (no downdraft, LFS below the LCL, UPDINC capped at 1.5, the
    negative-moisture borrow, FABE > 1, the iteration cap, calm winds).
    Which of them enter the fixture is decided by measured line coverage of
    the WRF source (select_cases.py), not by their names.
    """
    ztop = rng.uniform(15000.0, 22000.0)
    z_ifc = ztop * np.linspace(0.0, 1.0, nz + 1) ** rng.uniform(1.0, 1.6)
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)
    psfc = rng.uniform(78000.0, 103000.0)
    ts = rng.uniform(258.0, 312.0)
    zbl = rng.uniform(100.0, 3000.0)
    gbl = rng.uniform(2.0, 9.9) * 1.0e-3
    gamma = rng.uniform(3.0, 9.5) * 1.0e-3
    ztrop = rng.uniform(7000.0, 17500.0)
    t = np.where(z < zbl, ts - gbl * z, ts - gbl * zbl - gamma * (z - zbl))
    ttrop = ts - gbl * zbl - gamma * (ztrop - zbl)
    t = np.where(z > ztrop, ttrop + rng.uniform(-1.0, 3.0) * 1.0e-3
                 * (z - ztrop), t)
    for _ in range(int(rng.integers(0, 3))):
        t = t + rng.uniform(-4.0, 8.0) * np.exp(
            -((z - rng.uniform(300.0, 9000.0)) / rng.uniform(150.0, 900.0))
            ** 2)
    t = np.maximum(t, 180.0)
    knots = np.sort(rng.uniform(0.0, ztop, 5))
    rh = np.interp(z, knots, rng.uniform(0.0, 1.05, 5))
    rh = np.clip(rh + rng.normal(0.0, 0.05, nz), 0.003, 1.08)
    p = np.empty(nz)
    qv = np.zeros(nz)
    for _ in range(3):
        pk = psfc
        for k in range(nz):
            tv = t[k] * (1.0 + 0.608 * qv[k])
            half = dz[k] * 0.5
            p[k] = pk * np.exp(-G * half / (RD * tv))
            pk = p[k] * np.exp(-G * half / (RD * tv))
        qv = np.maximum(rh * _qsat(t, p), 1.0e-8)
    tv = t * (1.0 + 0.608 * qv)
    rho = p / (RD * tv) * (1.0 + qv)
    pi = (p / 1.0e5) ** (RD / CP)
    if rng.uniform() < 0.1:
        u = np.zeros(nz)
        v = np.zeros(nz)
    else:
        u = rng.uniform(-15.0, 15.0) + rng.uniform(-6e-3, 6e-3) * z
        v = rng.uniform(-15.0, 15.0) + rng.uniform(-6e-3, 6e-3) * z
    w = (rng.uniform(-0.3, 1.0) * np.exp(
        -((z_ifc - rng.uniform(200.0, 6000.0)) / rng.uniform(500.0, 4000.0))
        ** 2) + rng.normal(0.0, rng.uniform(0.0, 0.1), nz + 1))
    w[0] = 0.0
    w0avg = (0.5 * (w[:-1] + w[1:]) * rng.uniform(0.0, 1.5)
             + rng.normal(0.0, 0.01, nz))
    return dict(u=u, v=v, w=w, t=t, th=t / pi, qv=qv, dz8w=dz, p=p,
                rho=rho, pi=pi, w0avg=w0avg)


def wide_candidate(seed: int, index: int):
    """Candidate ``index`` of the wide family, reproducible on its own."""
    rng = np.random.default_rng([seed, 7, index])
    nz = int(rng.choice([40, 49, 60]))
    col = wide(rng, nz=nz)
    knobs = driver_knobs(rng, index=index)
    return (f"wide/nz{nz}/c{index}", col, knobs)


def real74(path, label, rng, *, perturb):
    with np.load(path, allow_pickle=False) as data:
        col = {name: data[f"{label}_{name}"].astype(np.float64)
               for name in ("u", "v", "temperature", "qv", "pressure",
                            "exner", "dz", "w")}
    nz = col["temperature"].size
    t = col["temperature"].copy()
    qv = col["qv"].copy()
    if perturb:
        t = t + rng.normal(0.0, 0.4, nz) * np.exp(-np.arange(nz) / 8.0)
        qv = qv * (1.0 + rng.normal(0.0, 0.05, nz))
    p = col["pressure"]
    tv = t * (1.0 + 0.608 * qv)
    rho = p / (RD * tv) * (1.0 + qv)
    pi = col["exner"]
    w_mass = col["w"]
    w = np.zeros(nz + 1)
    w[1:nz] = 0.5 * (w_mass[:-1] + w_mass[1:])
    w[nz] = w_mass[-1]
    w0avg = w_mass * rng.uniform(0.5, 1.5)
    return dict(u=col["u"], v=col["v"], w=w, t=t, th=t / pi, qv=qv,
                dz8w=col["dz"], p=p, rho=rho, pi=pi, w0avg=w0avg)


def driver_knobs(rng, *, index):
    dt = float(rng.choice([20.0, 30.0, 45.0, 60.0, 90.0, 150.0, 300.0,
                           600.0]))
    dx = float(rng.choice([6000.0, 9000.0, 12000.0, 20000.0, 30000.0]))
    stepcu = int(rng.choice([1, 1, 3, 5, 10]))
    cudt = float(rng.choice([0.0, 5.0, 10.0]))
    adapt = int(index % 9 == 4)
    species = list(SPECIES_SETS)[index % len(SPECIES_SETS)]
    edrates = int(index % 3 != 0)
    hold = index % 11
    if hold == 3:
        nca = 2.0 * dt          # held: NCA > 0.5*DT, the scheme skips it
    elif hold == 6:
        nca = 0.5 * dt          # boundary: NOT > 0.5*DT, so it is run
    elif hold == 8:
        nca = 0.3 * dt          # expiring timer below the hold threshold
    else:
        nca = -100.0            # kfinit's cold-start value
    return dict(dt=dt, dx=dx, stepcu=stepcu, cudt=cudt, adapt=adapt,
                species=species, kf_edrates=edrates, nca=nca)


def build_cases(real_path: Path | None, seed: int):
    rng = np.random.default_rng(seed)
    regimes = ["deep-moist", "deep-cold-top", "midlat", "high-terrain",
               "shallow", "stable", "dry", "cold-season"]
    cases = []
    index = 0
    for nz in (40, 49, 60):
        for rep in range(9):
            for regime in regimes:
                col = synthetic(rng, nz=nz, regime=regime)
                knobs = driver_knobs(rng, index=index)
                cases.append((f"{regime}/nz{nz}/r{rep}", col, knobs))
                index += 1
    if real_path is not None and real_path.exists():
        for label in ("unstable", "stable"):
            for rep in range(12):
                col = real74(real_path, label, rng, perturb=rep > 0)
                knobs = driver_knobs(rng, index=index)
                cases.append((f"real74-{label}/r{rep}", col, knobs))
                index += 1
    return cases


def write_cases(cases, path: Path, meta_path: Path) -> None:
    meta = []
    with path.open("wb") as fh:
        fh.write(struct.pack("<3i", MAGIC, VERSION, len(cases)))
        for name, col, knobs in cases:
            nz = col["t"].size
            warm, fqi, fqs, fqr = SPECIES_SETS[knobs["species"]]
            fh.write(struct.pack("<8i", nz, knobs["stepcu"],
                                 knobs["adapt"], warm, fqi, fqs, fqr,
                                 knobs["kf_edrates"]))
            fh.write(struct.pack("<4f", knobs["dt"], knobs["dx"],
                                 knobs["cudt"], knobs["nca"]))
            for field in FIELDS:
                values = np.asarray(col[field], dtype="<f4")
                expected = nz + 1 if field == "w" else nz
                assert values.shape == (expected,), (name, field)
                assert np.all(np.isfinite(values)), (name, field)
                fh.write(values.tobytes())
            meta.append({"name": name, "nz": nz, **knobs})
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")


#: The wide-family candidates select_cases.py kept because each one
#: reached WRF source lines no earlier case did.  Rewritten only by
#: select_cases.py.
SELECTED_WIDE = Path(__file__).with_name("selected_wide.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--real74", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--candidates", type=int, default=0,
                        help="write N wide candidates instead of the "
                             "fixture's case set (select_cases.py)")
    args = parser.parse_args()
    if args.candidates:
        cases = [wide_candidate(args.seed, i)
                 for i in range(args.candidates)]
    else:
        cases = build_cases(args.real74, args.seed)
        if SELECTED_WIDE.exists():
            chosen = json.loads(SELECTED_WIDE.read_text(encoding="utf-8"))
            cases += [wide_candidate(args.seed, i)
                      for i in chosen["candidates"]]
    write_cases(cases, args.out, args.out.with_suffix(".json"))
    print(f"{len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
