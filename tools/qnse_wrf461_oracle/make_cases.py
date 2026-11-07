"""Write the QNSE PBL oracle's input columns (one stream file per arm).

The columns are chosen to reach branches, not to look like a forecast:
dry convective land, stable land, stratocumulus-topped (land and sea),
shallow cumulus (land and sea), plain sea, and an edge group (snow, sea
ice over land, sub-freezing columns that take the mass-flux updraft's
cold ELSEWHERE arm, 1500 m and 3000 m terrain, a low-level jet, a weak
u*, a southern-hemisphere Coriolis parameter, TKE at the cold-start
floor, a nonzero countergradient CT).  Every array is float32 and is
written in Fortran index order, (ncol, nz) with the column index fastest.

    python make_cases.py OUT_DIR

writes ``cases-mf0.bin`` (mfshconv = 0, dt = 30 s, stepbl = 1) and
``cases-mf1.bin`` (mfshconv = 1, dt = 10 s, stepbl = 3), both 72 columns
by 40 levels, four steps each.  The same 72 columns feed both arms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

F = np.float32
NZ = 40
NSTEP = 4
RD, RV, CP, G = 287.0, 461.6, 7.0 * 287.0 / 2.0, 9.81

#: (arm, mfshconv, dt, stepbl)
ARMS = (("mf0", 0, 30.0, 1), ("mf1", 1, 10.0, 3))


def _dz() -> np.ndarray:
    dz = np.empty(NZ)
    for k in range(NZ):
        dz[k] = min(40.0 * 1.13 ** k, 650.0)
    return dz


def _column(kind: str, rng: np.random.Generator, variant: int) -> dict:
    """One column in float64, cast to float32 by the caller."""
    dz = _dz() * (1.0 + 0.04 * rng.standard_normal())
    zf = np.concatenate([[0.0], np.cumsum(dz)])          # interfaces AGL
    zm = 0.5 * (zf[:-1] + zf[1:])                       # mass levels AGL
    c = dict(ht=0.0, xland=1.0, sice=0.0, snow=0.0, ct=0.0,
             corf=1.0e-4, ustar=0.35, znt=0.1)
    psfc = 1.0e5 - 800.0 * rng.random()
    th = np.empty(NZ)
    qv = np.empty(NZ)
    qc = np.zeros(NZ)
    pbl = 1000.0
    tke_mode = "developed"
    if kind == "convective":
        pbl = 900.0 + 900.0 * rng.random()
        th0 = 298.0 + 6.0 * rng.random()
        sup = 1.0 + 2.5 * rng.random()
        th = np.where(zm < 60.0, th0 + sup * (1.0 - zm / 60.0), th0)
        th = np.where(zm > pbl, th0 + 1.0 + 0.0045 * (zm - pbl), th)
        q0 = 0.006 + 0.004 * rng.random()
        qv = np.where(zm < pbl, q0, q0 * np.exp(-(zm - pbl) / 1800.0) * 0.6)
        c.update(hfx=150.0 + 300.0 * rng.random(),
                 lh=40.0 + 200.0 * rng.random(),
                 ustar=0.25 + 0.4 * rng.random(), akhs=0.02 + 0.03 * rng.random(),
                 akms=0.02 + 0.02 * rng.random(), chklowq=0.3 + 0.7 * rng.random())
    elif kind == "stable":
        pbl = 150.0 + 200.0 * rng.random()
        th0 = 283.0 + 8.0 * rng.random()
        th = th0 + np.where(zm < pbl, 0.02 * zm, 0.02 * pbl + 0.004 * (zm - pbl))
        qv = 0.005 * np.exp(-zm / 2500.0) * (0.7 + 0.3 * rng.random())
        tke_mode = "weak"
        c.update(hfx=-(5.0 + 55.0 * rng.random()), lh=20.0 * rng.random(),
                 ustar=0.06 + 0.2 * rng.random(), akhs=0.002 + 0.006 * rng.random(),
                 akms=0.003 + 0.008 * rng.random(), chklowq=0.2 + 0.6 * rng.random())
    elif kind in ("stratocumulus", "stratocumulus-sea"):
        pbl = 700.0 + 500.0 * rng.random()
        th0 = 288.0 + 4.0 * rng.random()
        th = np.where(zm < pbl, th0, th0 + 7.0 + 0.006 * (zm - pbl))
        qt = 0.0085 + 0.0015 * rng.random()
        qv = np.where(zm < pbl, qt, 0.002 * np.exp(-(zm - pbl) / 2000.0))
        cloud = (zm < pbl) & (zm > pbl - 250.0 - 150.0 * rng.random())
        qc = np.where(cloud, 1.0e-4 + 4.0e-4 * (zm - (pbl - 400.0)).clip(0) / 400.0, 0.0)
        qv = np.where(cloud, qv - qc, qv)
        c.update(hfx=5.0 + 30.0 * rng.random(), lh=40.0 + 80.0 * rng.random(),
                 ustar=0.2 + 0.2 * rng.random(), akhs=0.008 + 0.01 * rng.random(),
                 akms=0.008 + 0.01 * rng.random(), chklowq=0.6 + 0.4 * rng.random())
        if kind == "stratocumulus-sea":
            c["xland"] = 2.0
            c["chklowq"] = 1.0
    elif kind in ("shallow-cumulus", "shallow-cumulus-sea"):
        pbl = 500.0 + 300.0 * rng.random()
        th0 = 297.0 + 3.0 * rng.random()
        th = np.where(zm < 50.0, th0 + 0.8 * (1.0 - zm / 50.0), th0)
        th = np.where(zm > pbl, th0 + 0.5 + 0.0038 * (zm - pbl), th)
        q0 = 0.0150 + 0.0035 * rng.random()
        qv = np.where(zm < pbl, q0 - 1.0e-6 * zm,
                      (q0 - 0.002) * np.exp(-(zm - pbl) / 2600.0))
        c.update(hfx=15.0 + 140.0 * rng.random(), lh=150.0 + 250.0 * rng.random(),
                 ustar=0.2 + 0.3 * rng.random(), akhs=0.012 + 0.02 * rng.random(),
                 akms=0.012 + 0.02 * rng.random(), chklowq=0.7 + 0.3 * rng.random())
        if kind == "shallow-cumulus-sea":
            c["xland"] = 2.0
            c["chklowq"] = 1.0
    elif kind == "sea":
        pbl = 400.0 + 600.0 * rng.random()
        th0 = 290.0 + 10.0 * rng.random()
        lapse = 0.003 + 0.004 * rng.random()
        th = np.where(zm < pbl, th0 + 0.2 * rng.standard_normal(), th0 + 1.5 + lapse * (zm - pbl))
        qv = 0.004 + 0.008 * rng.random() * np.exp(-zm / 2000.0)
        c.update(xland=2.0, hfx=-20.0 + 80.0 * rng.random(), lh=20.0 + 200.0 * rng.random(),
                 ustar=0.1 + 0.4 * rng.random(), akhs=0.004 + 0.02 * rng.random(),
                 akms=0.004 + 0.02 * rng.random(), chklowq=1.0)
    elif kind == "edge":
        th0 = 290.0
        pbl = 800.0
        th = np.where(zm < pbl, th0, th0 + 1.0 + 0.004 * (zm - pbl))
        qv = 0.006 * np.exp(-zm / 2500.0)
        c.update(hfx=120.0, lh=80.0, ustar=0.3, akhs=0.02, akms=0.02, chklowq=0.8)
        if variant == 0:   # snow on land
            c.update(snow=0.05, hfx=-30.0, lh=5.0)
            th = th0 - 15.0 + np.where(zm < 200.0, 0.03 * zm, 6.0 + 0.004 * (zm - 200.0))
            tke_mode = "weak"
        elif variant == 1:  # sea ice flagged on a land point
            c.update(sice=0.9, hfx=-15.0, lh=2.0)
            th = th0 - 25.0 + 0.008 * zm
            qv = 0.0015 * np.exp(-zm / 2500.0)
            tke_mode = "weak"
        elif variant == 2:  # sub-freezing convective: MF updraft's cold arm
            th = np.where(zm < 50.0, 262.0 + 2.0 * (1.0 - zm / 50.0), 262.0)
            th = np.where(zm > 700.0, 263.0 + 0.004 * (zm - 700.0), th)
            qv = np.where(zm < 700.0, 0.0018, 0.0012 * np.exp(-(zm - 700.0) / 1500.0))
            c.update(hfx=180.0, lh=30.0)
        elif variant == 3:  # 1500 m terrain
            c.update(ht=1500.0)
            psfc = 8.45e4
        elif variant == 4:  # 3000 m terrain, convective
            c.update(ht=3000.0, hfx=300.0)
            psfc = 7.0e4
            th = np.where(zm < 60.0, 304.0 + 2.0 * (1.0 - zm / 60.0), 304.0)
            th = np.where(zm > 1500.0, 305.0 + 0.004 * (zm - 1500.0), th)
        elif variant == 5:  # low-level jet over a stable layer
            th = th0 + np.where(zm < 300.0, 0.015 * zm, 4.5 + 0.004 * (zm - 300.0))
            c.update(hfx=-20.0, lh=5.0, ustar=0.25, akhs=0.004, akms=0.006)
        elif variant == 6:  # weak u*
            c.update(ustar=0.012, hfx=-3.0, lh=1.0, akhs=0.0015, akms=0.002)
            th = th0 + 0.012 * zm
            tke_mode = "weak"
        elif variant == 7:  # southern hemisphere
            c.update(corf=-7.0e-5)
        elif variant == 8:  # TKE at the cold-start floor everywhere
            tke_mode = "floor"
        elif variant == 9:  # countergradient CT carried in
            c.update(ct=0.35)
            th = np.where(zm < 600.0, th0 + 0.3 * np.sin(zm / 80.0), th)
        elif variant == 10:  # equatorial: corf below EPS1
            c.update(corf=0.0)
        else:              # deep, moist, nearly saturated warm column
            th = np.where(zm < 1800.0, 301.0, 302.0 + 0.003 * (zm - 1800.0))
            qv = np.where(zm < 1800.0, 0.0185, 0.012 * np.exp(-(zm - 1800.0) / 2500.0))
            c.update(hfx=90.0, lh=380.0, xland=2.0, chklowq=1.0)
    else:
        raise ValueError(kind)
    qv = np.maximum(qv, 1.0e-6)
    # Hydrostatic pressure on the interfaces from the surface up.
    pint = np.empty(NZ + 1)
    pint[0] = psfc
    for k in range(NZ):
        tv = th[k] * (pint[k] / 1.0e5) ** (RD / CP) * (1.0 + 0.608 * qv[k])
        pint[k + 1] = pint[k] * np.exp(-G * dz[k] / (RD * tv))
    pmid = 0.5 * (pint[:-1] + pint[1:])
    # Winds: an Ekman-ish turning profile, with a jet for edge variant 5.
    speed = 2.0 + 10.0 * rng.random()
    turn = np.deg2rad(25.0 + 20.0 * rng.random())
    shape = np.log1p(zm / 10.0) / np.log1p(pbl / 10.0)
    shape = np.minimum(shape, 1.0) + 0.0015 * np.maximum(zm - pbl, 0.0)
    ang = turn * (1.0 - np.minimum(zm / pbl, 1.0))
    u = speed * shape * np.cos(ang)
    v = speed * shape * np.sin(ang) + 0.3 * rng.standard_normal()
    if kind == "edge" and variant == 5:
        u = u + 12.0 * np.exp(-((zm - 350.0) / 150.0) ** 2)
    if kind == "edge" and variant == 6:
        u = 0.3 * u
        v = 0.3 * v
    # TKE (WRF's TKE_PBL = q**2/2).
    if tke_mode == "floor":
        tke = np.full(NZ, 0.01)                      # QNSEPBLINIT's EPSQ2L
    elif tke_mode == "weak":
        tke = np.where(zm < pbl, 0.05 + 0.15 * (1.0 - zm / pbl), 0.005)
    else:
        tke = np.where(zm < pbl, 0.4 + 1.2 * (1.0 - zm / pbl) * rng.random() + 0.3,
                       0.005)
    exner = (pmid / 1.0e5) ** (RD / CP)
    t = th * exner
    rho = pmid / (RD * t * (1.0 + 0.608 * qv))
    tsk = (t[0] + (c["hfx"] / 40.0)) if c["xland"] < 1.5 else t[0] + 0.5 * rng.standard_normal()
    thz0 = th[0] + c["hfx"] / 120.0
    qsfc = qv[0] * (1.0 + 0.1 * rng.random())
    qz0 = 0.5 * (qv[0] + qsfc)
    qfx = c["lh"] / 2.5e6
    c.update(dz=dz, pint=pint, pmid=pmid, th=th, qv=qv, qc=qc, u=u, v=v,
             tke=tke, exner=exner, t=t, rho=rho, tsk=tsk, thz0=thz0,
             qsfc=qsfc, qz0=qz0, qfx=qfx,
             uz0=0.05 * rng.standard_normal(), vz0=0.05 * rng.standard_normal())
    return c


#: 72 columns: 12 convective, 10 stable, 6+6 stratocumulus, 8+8 shallow
#: cumulus, 10 sea, 12 edge.
PLAN = (("convective", 12), ("stable", 10), ("stratocumulus", 6),
        ("stratocumulus-sea", 6), ("shallow-cumulus", 8),
        ("shallow-cumulus-sea", 8), ("sea", 10), ("edge", 12))


def build_columns(seed: int = 4146) -> list[dict]:
    rng = np.random.default_rng(seed)
    cols = []
    for kind, count in PLAN:
        for variant in range(count):
            cols.append(_column(kind, rng, variant))
    return cols


SURFACE = ("ht", "tsk", "qsfc", "chklowq", "thz0", "qz0", "uz0", "vz0",
           "corf", "xland", "sice", "snow", "ustar", "znt", "ct", "akhs",
           "akms", "lh", "hfx", "qfx")
LEVELS = ("dz", "pmid", "th", "t", "exner", "qv", "qc", "u", "v", "rho",
          "tke")


def write_arm(path: Path, cols: list[dict], mfsh: int, dt: float,
              stepbl: int) -> None:
    ncol = len(cols)
    with path.open("wb") as out:
        np.array([ncol, NZ, NSTEP, mfsh, stepbl], dtype=np.int32).tofile(out)
        np.array([dt], dtype=F).tofile(out)
        for name in SURFACE:
            np.array([c[name] for c in cols], dtype=F).tofile(out)
        for name in LEVELS:
            # Fortran (ncol, nz): column index fastest == C order (nz, ncol).
            np.ascontiguousarray(
                np.stack([c[name] for c in cols], axis=1).astype(F)).tofile(out)
        np.ascontiguousarray(
            np.stack([c["pint"] for c in cols], axis=1).astype(F)).tofile(out)


def main(argv: list[str]) -> int:
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cols = build_columns()
    for arm, mfsh, dt, stepbl in ARMS:
        write_arm(out / f"cases-{arm}.bin", cols, mfsh, dt, stepbl)
    print(f"wrote {len(cols)} columns x {NZ} levels for arms "
          + ", ".join(a[0] for a in ARMS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
