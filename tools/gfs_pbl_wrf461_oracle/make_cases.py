"""Generate the GFS PBL oracle's input columns (bl_pbl_physics=3).

Seventy-two columns per vertical grid: eleven variants each of six regime
families -- convective daytime land, stable night, neutral windy,
cloud-topped, very shallow PBL, and a deep PBL that reaches the scheme's
search ceiling (KMPBL = KM/2), including a separate
lid-focused grid -- plus six edge columns aimed at the scheme's branch
points (BR exactly
zero, zero surface buoyancy flux, a buoyancy flux carried by evaporation
alone, a calm wind, an extreme heat flux, vapour at zero aloft).

The columns are hydrostatically built (interfaces from a stretched height
grid, pressure integrated with the layer virtual temperature) and the
surface-layer words the scheme reads are mutually consistent: GZ1OZ0 is
ln(za/z0) at the first mass level, PSIM/PSIH follow Businger-Dyer forms of
z/L, UST is the neutral-corrected friction velocity of WSPD.  This is input
plausibility, not a surface layer: the oracle grades arithmetic, so every
word only has to be one WRF can be handed.

Usage: python make_cases.py <out_dir>
Writes cases-<grid>-<config>.in (the oracle's stream format) and
cases-index.json naming every column's regime and parameters.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

F = np.float32
R_D, R_V, G, CP = 287.0, 461.6, 9.81, 1004.5
RCP = F(F(287.0) / F(1004.5))
KARMAN = 0.4

#: name -> (levels, model-top height AGL m, stretching exponent).
GRIDS = {
    "g40": (40, 20000.0, 4.0),
    "g61": (61, 25000.0, 4.5),
    # Low top: KMPBL = 8 sits below 2.2 km, so the deep family's search
    # stops at the ceiling.
    "g16": (16, 6500.0, 1.2),
    # A lid-focused grid: its upper half is clustered near the 3 km lid,
    # so the KM/2 search ceiling can represent a PBL near the physical top.
    "g16top": (16, 3000.0, (25.0, 0.98)),
}

#: name -> (step seconds, steps, ice carried).
CONFIGS = {
    "ice60": (60.0, 4, 1),
    "noice180": (180.0, 4, 0),
}

FAMILIES = ("convective", "stable", "neutral_windy", "cloud_topped",
            "very_shallow", "deep_to_ceiling")


def interfaces(nz: int, top: float,
               a: float | tuple[float, float]) -> np.ndarray:
    if isinstance(a, tuple):
        # Table metadata: first layer depth, fraction of the lid reached
        # by the lower half. Keep a low surface level while testing the
        # search ceiling close to the physical lid.
        first_depth, lower_fraction = a
        half = nz // 2
        lower = np.r_[0.0, np.linspace(first_depth, top * lower_fraction, half)]
        upper = np.linspace(top * lower_fraction, top, nz - half + 1)[1:]
        return np.r_[lower, upper]
    s = np.arange(nz + 1) / nz
    return top * (np.exp(a * s) - 1.0) / (np.exp(a) - 1.0)


def stability(br: float, gz: float) -> tuple[float, float]:
    """Businger-Dyer psim/psih at z/L estimated from BR."""
    if br >= 0.0:
        zol = min(br * gz / max(1.0 - 5.0 * min(br, 0.19), 0.05), 10.0)
        psim = max(-5.0 * zol, -10.0)
        return psim, psim
    zol = max(br * gz, -10.0)
    x = (1.0 - 16.0 * zol) ** 0.25
    psim = (2.0 * math.log((1.0 + x) / 2.0) + math.log((1.0 + x * x) / 2.0)
            - 2.0 * math.atan(x) + math.pi / 2.0)
    psih = 2.0 * math.log((1.0 + x * x) / 2.0)
    return psim, psih


def column(family: str, variant: int, zi: np.ndarray, rng) -> dict:
    nz = zi.size - 1
    zm = 0.5 * (zi[:-1] + zi[1:])
    th0 = rng.uniform(286.0, 308.0)
    qv0 = rng.uniform(4e-3, 1.6e-2)
    ht = float(rng.choice([0.0, 250.0, 900.0, 1600.0]))
    psfc = 101325.0 * math.exp(-ht / 8000.0) * rng.uniform(0.985, 1.01)
    z0 = float(rng.choice([2e-4, 0.01, 0.1, 0.5, 1.0]))
    ug = rng.uniform(2.0, 10.0)
    veer = math.radians(rng.uniform(-20.0, 40.0))
    hfx = qfx = 0.0
    br = 0.0
    h = 1000.0
    th = np.full(nz, th0)
    qv = np.empty(nz)
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    jet = None
    if family == "convective":
        h = rng.uniform(700.0, 2600.0)
        jump = rng.uniform(1.0, 5.0)
        th = th0 - 1.2 * np.exp(-zm / 40.0) + np.where(
            zm > h, jump + rng.uniform(3.0, 6.0) * (zm - h) / 1000.0, 0.0)
        hfx = rng.uniform(80.0, 450.0)
        qfx = rng.uniform(2e-5, 2e-4)
        br = -rng.uniform(0.2, 6.0)
        if variant % 3 == 0:
            # A superadiabatic layer aloft: Ri < 0 above the PBL.
            lay = (zm > h + 1500.0) & (zm < h + 2300.0)
            th = th - np.where(lay, 0.004 * (zm - h - 1500.0), 0.0) \
                - np.where(zm >= h + 2300.0, 3.2, 0.0)
    elif family == "stable":
        hs = rng.uniform(60.0, 300.0)
        th = th0 + rng.uniform(4.0, 12.0) * (1.0 - np.exp(-zm / hs)) \
            + 3.5 * zm / 1000.0
        hfx = -rng.uniform(5.0, 60.0)
        qfx = rng.uniform(0.0, 1e-5)
        br = rng.uniform(0.02, 1.2)
        ug = rng.uniform(0.5, 4.0)
        if variant % 2 == 0:
            jet = (rng.uniform(150.0, 450.0), rng.uniform(6.0, 14.0))
        h = hs
    elif family == "neutral_windy":
        th = th0 + rng.uniform(0.3, 1.5) * zm / 1000.0 + np.where(
            zm > 1500.0, 3.0 * (zm - 1500.0) / 1000.0, 0.0)
        hfx = rng.uniform(-8.0, 8.0)
        qfx = rng.uniform(0.0, 3e-5)
        br = rng.uniform(-0.01, 0.01)
        ug = rng.uniform(12.0, 26.0)
        h = 1500.0
    elif family == "cloud_topped":
        h = rng.uniform(600.0, 1600.0)
        th = th0 + np.where(zm > h, rng.uniform(5.0, 10.0)
                            + 5.0 * (zm - h) / 1000.0, 0.0)
        depth = rng.uniform(150.0, 450.0)
        cloud = (zm > h - depth) & (zm <= h)
        qc = np.where(cloud, rng.uniform(5e-5, 6e-4)
                      * (zm - (h - depth)) / depth, 0.0)
        cold = (zm > 4500.0) & (zm < 7500.0)
        qi = np.where(cold, rng.uniform(1e-6, 8e-5), 0.0)
        qc = qc + np.where(cold & (zm < 5200.0), 2e-5, 0.0)
        hfx = rng.uniform(10.0, 90.0)
        qfx = rng.uniform(1e-5, 8e-5)
        br = -rng.uniform(0.01, 0.8)
    elif family == "very_shallow":
        hs = rng.uniform(25.0, 90.0)
        th = th0 + rng.uniform(40.0, 110.0) * np.minimum(zm, 4.0 * hs) \
            / 1000.0 + 3.0 * zm / 1000.0
        hfx = rng.choice([-rng.uniform(1.0, 15.0), rng.uniform(0.5, 4.0)])
        qfx = rng.uniform(0.0, 5e-6)
        br = rng.uniform(0.15, 3.0) if hfx < 0 else -rng.uniform(0.001, 0.05)
        ug = rng.uniform(0.3, 3.0)
        h = hs
    elif family == "deep_to_ceiling":
        h = rng.uniform(2800.0, 4800.0)
        th = th0 + 8.0 - 2.0 * np.exp(-zm / 60.0) + np.where(
            zm > h, rng.uniform(1.0, 3.0) + 4.0 * (zm - h) / 1000.0, 0.0)
        hfx = rng.uniform(250.0, 650.0)
        qfx = rng.uniform(0.0, 6e-5)
        br = -rng.uniform(2.0, 30.0)
        qv0 = rng.uniform(1e-3, 6e-3)
        ug = rng.uniform(1.0, 6.0)
    else:
        raise ValueError(family)
    # Moisture: mixed below h, decaying above, vapour exactly zero at the
    # very top on some variants (the scheme's QMIN floor).
    qv = np.where(zm <= h, qv0, qv0 * np.exp(-(zm - h) / 1800.0))
    qv = np.maximum(qv, 2e-6)
    if variant % 4 == 1:
        qv[-3:] = 0.0
    # Winds: log profile to a reference height, veering with height, an
    # optional low-level jet.
    zr = max(h, 200.0)
    spd = ug * np.clip(np.log((zm + z0) / z0) / np.log((zr + z0) / z0),
                       0.05, None)
    spd = np.where(zm > zr, ug + (zm - zr) * rng.uniform(0.0, 3.0e-3), spd)
    if jet is not None:
        spd = spd + jet[1] * np.exp(-((zm - jet[0]) / (0.5 * jet[0])) ** 2)
    ang = veer * np.minimum(zm / zr, 1.0) + rng.uniform(0.0, 2.0 * math.pi)
    u = spd * np.cos(ang)
    v = spd * np.sin(ang)
    return dict(th=th, qv=qv, qc=qc, qi=qi, u=u, v=v, ht=ht, psfc=psfc,
                z0=z0, hfx=hfx, qfx=qfx, br=br, h=h)


def edge(index: int, zi: np.ndarray, rng) -> dict:
    """Six branch-point columns built on a convective base."""
    c = column("convective", 7, zi, rng)
    if index == 0:
        c["br"] = 0.0                 # SFCFLG stays true at BR == 0
    elif index == 1:
        c["hfx"], c["qfx"] = 0.0, 0.0  # zero buoyancy flux: PBLFLG off
    elif index == 2:
        c["hfx"], c["qfx"] = -2.0, 3e-4  # buoyancy flux from evaporation
    elif index == 3:
        c["u"] = c["u"] * 1e-3        # calm: SPD1 at its floor
        c["v"] = c["v"] * 1e-3
    elif index == 4:
        c["hfx"], c["br"] = 900.0, -60.0  # GAMCRT caps HGAMT
    elif index == 5:
        c["qv"] = c["qv"].copy()
        c["qv"][len(c["qv"]) // 2:] = 0.0  # QMIN floor through half a column
        c["br"] = 0.5                 # stable surface over a dry column
        c["hfx"] = -10.0
    c["family"] = f"edge{index}"
    return c


def build(nz: int, top: float, a: float, seed: int):
    zi_agl = interfaces(nz, top, a)
    rng = np.random.default_rng(seed)
    cols, meta = [], []
    for fam in FAMILIES:
        for variant in range(11):
            c = column(fam, variant, zi_agl, rng)
            c["family"] = fam
            cols.append(c)
    for index in range(6):
        cols.append(edge(index, zi_agl, rng))
    ncol = len(cols)
    kme = nz + 1
    shape = (kme, ncol)  # transposed to Fortran (ncol, kme) on write
    out = {name: np.zeros(shape, F) for name in
           ("u", "v", "th", "t", "qv", "qc", "qi", "p", "pi", "dz8w", "z")}
    sfc = {name: np.zeros(ncol, F) for name in
           ("psfc", "ust", "pbl", "psim", "psih", "hfx", "qfx", "tsk",
            "gz1oz0", "wspd", "br")}
    for i, c in enumerate(cols):
        # Hydrostatic interfaces: p(k+1) = p(k) exp(-g dz / (Rd Tv)).
        pint = np.empty(nz + 1)
        pint[0] = c["psfc"]
        pmid = np.empty(nz)
        for k in range(nz):
            dz = zi_agl[k + 1] - zi_agl[k]
            pm = pint[k]
            for _ in range(3):
                tv = c["th"][k] * (pm / 1e5) ** (R_D / CP) \
                    * (1.0 + 0.608 * c["qv"][k])
                pint[k + 1] = pint[k] * math.exp(-G * dz / (R_D * tv))
                pm = math.sqrt(pint[k] * pint[k + 1])
            pmid[k] = pm
        zi32 = F(c["ht"]) + zi_agl.astype(F)
        z32 = F(0.5) * (zi32[:-1] + zi32[1:])
        p32 = pmid.astype(F)
        pi32 = (p32 / F(1e5)) ** RCP
        th32 = c["th"].astype(F)
        out["u"][:nz, i] = c["u"]
        out["v"][:nz, i] = c["v"]
        out["th"][:nz, i] = th32
        out["t"][:nz, i] = th32 * pi32
        out["qv"][:nz, i] = c["qv"]
        out["qc"][:nz, i] = c["qc"]
        out["qi"][:nz, i] = c["qi"]
        out["p"][:nz, i] = p32
        out["pi"][:nz, i] = pi32
        out["dz8w"][:nz, i] = zi32[1:] - zi32[:-1]
        out["z"][:nz, i] = z32
        out["z"][nz, i] = zi32[nz]
        za = float(z32[0] - zi32[0])
        gz = math.log(za / c["z0"])
        psim, psih = stability(c["br"], gz)
        wspd = max(math.hypot(c["u"][0], c["v"][0]), 0.1)
        if c["br"] < 0.0:
            wspd = math.sqrt(wspd * wspd + 1.0)
        sfc["psfc"][i] = c["psfc"]
        sfc["ust"][i] = KARMAN * wspd / max(gz - psim, 0.5)
        sfc["pbl"][i] = c["h"]
        sfc["psim"][i] = psim
        sfc["psih"][i] = psih
        sfc["hfx"][i] = c["hfx"]
        sfc["qfx"][i] = c["qfx"]
        sfc["tsk"][i] = c["th"][0] * (c["psfc"] / 1e5) ** (R_D / CP) \
            + (2.0 if c["hfx"] > 0 else -2.0)
        sfc["gz1oz0"][i] = gz
        sfc["wspd"][i] = wspd
        sfc["br"][i] = c["br"]
        meta.append({"column": i, "family": c["family"],
                     "ht": c["ht"], "z0": c["z0"], "h": round(c["h"], 1),
                     "hfx": round(float(c["hfx"]), 3),
                     "br": round(float(c["br"]), 5)})
    return out, sfc, meta


def write_case(path: Path, out, sfc, nstep: int, flqi: int, dt: float):
    nz = out["u"].shape[0] - 1
    ncol = out["u"].shape[1]
    with open(path, "wb") as fh:
        fh.write(np.array([ncol, nz, nstep, flqi], "<i4").tobytes())
        fh.write(np.array([dt], "<f4").tobytes())
        for name in ("u", "v", "th", "t", "qv", "qc", "qi", "p", "pi",
                     "dz8w", "z"):
            # (kme, ncol) C order == Fortran (ncol, kme).
            fh.write(np.ascontiguousarray(out[name], "<f4").tobytes())
        for name in ("psfc", "ust", "pbl", "psim", "psih", "hfx", "qfx",
                     "tsk", "gz1oz0", "wspd", "br"):
            fh.write(np.ascontiguousarray(sfc[name], "<f4").tobytes())


def main(dest: str) -> None:
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    index = {}
    for gi, (grid, (nz, top, a)) in enumerate(GRIDS.items()):
        out, sfc, meta = build(nz, top, a, seed=20261008 + gi)
        index[grid] = {"nz": nz, "top_agl_m": top, "columns": meta}
        for cname, (dt, nstep, flqi) in CONFIGS.items():
            write_case(root / f"cases-{grid}-{cname}.in", out, sfc, nstep,
                       flqi, dt)
    (root / "cases-index.json").write_text(
        json.dumps({"grids": index, "configs": {
            k: {"dt": v[0], "steps": v[1], "ice": v[2]}
            for k, v in CONFIGS.items()}}, indent=1) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
