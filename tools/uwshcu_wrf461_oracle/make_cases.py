"""Column cases for the WRF v4.6.1 UW shallow cumulus (shcu_physics=2) oracle.

Writes one case file per vertical grid for run_uwshcu.F90 (the full WRF
driver, camuwshcu_driver, over several consecutive steps) and one r8 case
file per grid for run_uwshcu_core.F90 (compute_uwshcu_inv alone).

Driver case file ('UWSC', version 1): int32 magic, version, ncol, nk,
nsteps, has_qnc, has_qni, then float32 dt, then float32 arrays in the order
DRIVER_MASS (nk, ncol), DRIVER_FULL (nk+1, ncol), DRIVER_SFC (ncol).

Core case file ('UWS8', version 1): int32 magic, version, ncol, nk, then
float64 arrays in CAM orientation (index 1 = model top, as
compute_uwshcu_inv receives them): CORE_MASS (nk, ncol), CORE_FULL
(nk+1, ncol), CORE_TRACER (nk, 5, ncol), CORE_SFC (ncol), then float64 dt.

Regime families (each with its own seeded perturbations, so the same
command always writes the same bytes; the files are the fixture of record):

  trade       marine trade-wind cumulus: shallow mixed layer, conditionally
              unstable cloud layer, trade inversion
  continental daytime continental boundary layer with fair-weather cumulus
  mixedphase  cold-air outbreak over open water: cloud layer between 238
              and 268 K, so the condensate splits into liquid and ice
  dry         desert boundary layer, too dry to condense below 300 hPa or
              to condense at all
  highterrain elevated terrain (surface pressure 600-750 hPa)
  capped      a strong inversion right above the PBL top (large CIN)
  overshoot   deep conditional instability with moist free troposphere, so
              the updraft penetrates far above the PBL
  edge        hand-built edge columns: PBL top below the first interface,
              PBL top above the model, zero TKE, saturated source air,
              very dry source air, cold column

Usage:
  python make_cases.py OUTDIR [--per-family N] [--nsteps S] [--seed X]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DRIVER_MAGIC = 0x55575343   # 'UWSC'
CORE_MAGIC = 0x55575338     # 'UWS8'
R_D = 287.0
CP = 7.0 * R_D / 2.0
RCP = R_D / CP
G = 9.81
P0 = 1.0e5
EP = 0.622

FAMILIES = ("trade", "continental", "mixedphase", "dry", "highterrain",
            "capped", "overshoot")

#: Driver inputs, in the order run_uwshcu.F90 reads them.
DRIVER_MASS = ("u", "v", "t", "qv", "qc", "qi", "qnc", "qni", "p", "z",
               "pi", "cldfra", "cldfra_old", "dlf", "evapcsh", "icwmrsh",
               "rprdsh", "shfrc3d")
DRIVER_FULL = ("p8w", "z_at_w", "tke_pbl", "cmfmc", "cmfmc2", "cmfsl",
               "cmflq")
DRIVER_SFC = ("pblh", "ht", "cush", "cutop", "cubot", "rliq", "rliq2",
              "pratesh")

#: compute_uwshcu_inv inputs (r8, CAM orientation), in read order.
CORE_MASS = ("p0", "z0", "dp0", "dpdry0", "u0", "v0", "qv0", "ql0", "qi0",
             "t0", "s0", "cldfrct", "concldfrct")
CORE_FULL = ("ps0", "zs0", "tke")
CORE_SFC = ("pblh", "cush")


def interface_heights(nk: int, ztop: float, dz0: float) -> np.ndarray:
    """nk+1 interface heights above ground, stretched from dz0 to ztop."""
    dz = np.empty(nk)
    d = dz0
    for k in range(nk):
        dz[k] = d
        d = min(d * 1.12, 1100.0)
    z = np.concatenate([[0.0], np.cumsum(dz)])
    return z * (ztop / z[-1])


def esat_water(t):
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def regime(rng, family):
    """Parameters of one sounding: (dict)."""
    r = dict(ht=0.0, psfc=101500.0)
    u = rng.uniform
    if family == "trade":
        r.update(psfc=u(101000, 101800), th_ml=u(297.5, 300.0),
                 zi=u(450, 750), q_ml=u(0.0150, 0.0175),
                 g_cloud=u(3.0e-3, 4.2e-3), z_inv=u(1500, 2300),
                 dth_inv=u(2.0, 6.0), q_top=u(0.0080, 0.0110),
                 dq_inv=u(0.003, 0.006), g_free=u(3.5e-3, 4.5e-3),
                 hq=u(2000, 3000), tke0=u(0.3, 1.0), wind=u(-9, -5),
                 shear=u(-1e-3, 2e-3), cloud=True)
    elif family == "continental":
        r.update(psfc=u(95500, 98500), ht=u(200, 450), th_ml=u(302, 308),
                 zi=u(1100, 2100), q_ml=u(0.0100, 0.0145),
                 g_cloud=u(3.5e-3, 5.0e-3), z_inv=u(2600, 3600),
                 dth_inv=u(1.0, 4.0), q_top=u(0.0050, 0.0080),
                 dq_inv=u(0.001, 0.003), g_free=u(4.0e-3, 5.5e-3),
                 hq=u(2000, 3000), tke0=u(0.8, 2.2), wind=u(2, 8),
                 shear=u(0, 3e-3), cloud=True)
    elif family == "mixedphase":
        r.update(psfc=u(99000, 101500), th_ml=u(266, 274), zi=u(500, 1100),
                 q_ml=u(0.0020, 0.0034), g_cloud=u(3.0e-3, 4.5e-3),
                 z_inv=u(1600, 2800), dth_inv=u(1.5, 5.0),
                 q_top=u(0.0012, 0.0020), dq_inv=u(0.0003, 0.0008),
                 g_free=u(4.0e-3, 6.0e-3), hq=u(1500, 2500),
                 tke0=u(0.5, 1.8), wind=u(8, 16), shear=u(-2e-3, 1e-3),
                 cloud=True, cold=True)
    elif family == "dry":
        r.update(psfc=u(94000, 99000), ht=u(100, 800), th_ml=u(306, 316),
                 zi=u(1500, 3500), q_ml=u(0.0004, 0.0030),
                 g_cloud=u(5.0e-3, 7.0e-3), z_inv=u(3600, 4500),
                 dth_inv=u(0.5, 2.0), q_top=u(0.0002, 0.0012),
                 dq_inv=u(0.0001, 0.0005), g_free=u(5.0e-3, 7.0e-3),
                 hq=u(1500, 2500), tke0=u(1.0, 3.0), wind=u(0, 6),
                 shear=u(0, 2e-3), cloud=False)
    elif family == "highterrain":
        r.update(psfc=u(60000, 75000), ht=u(2400, 3800), th_ml=u(306, 314),
                 zi=u(700, 1600), q_ml=u(0.0050, 0.0090),
                 g_cloud=u(3.5e-3, 5.0e-3), z_inv=u(2000, 3000),
                 dth_inv=u(1.0, 3.0), q_top=u(0.0025, 0.0045),
                 dq_inv=u(0.0005, 0.0015), g_free=u(4.5e-3, 6.0e-3),
                 hq=u(1500, 2500), tke0=u(0.6, 2.0), wind=u(4, 12),
                 shear=u(0, 3e-3), cloud=True)
    elif family == "capped":
        r.update(psfc=u(99000, 101500), th_ml=u(296, 302), zi=u(500, 1000),
                 q_ml=u(0.0120, 0.0160), g_cloud=u(3.0e-3, 4.0e-3),
                 z_inv=None, dth_inv=u(6.0, 12.0), q_top=u(0.0030, 0.0060),
                 dq_inv=u(0.004, 0.008), g_free=u(4.0e-3, 5.0e-3),
                 hq=u(1500, 2500), tke0=u(0.2, 1.2), wind=u(-6, 6),
                 shear=u(-1e-3, 1e-3), cloud=True)
    elif family == "overshoot":
        r.update(psfc=u(99500, 101500), th_ml=u(300, 303), zi=u(500, 900),
                 q_ml=u(0.0170, 0.0200), g_cloud=u(2.0e-3, 3.0e-3),
                 z_inv=u(7000, 11000), dth_inv=u(0.0, 1.0),
                 q_top=u(0.0040, 0.0070), dq_inv=u(0.0, 0.001),
                 g_free=u(3.0e-3, 4.0e-3), hq=u(3000, 4500),
                 tke0=u(1.0, 3.0), wind=u(-4, 4), shear=u(-1e-3, 1e-3),
                 cloud=True)
    else:
        raise ValueError(family)
    return r


def sounding(rng, r, nk, ztop):
    """Profiles on nk mass levels from regime parameters r (float64)."""
    zi_w = interface_heights(nk, ztop, dz0=rng.uniform(25.0, 60.0))
    zm = 0.5 * (zi_w[1:] + zi_w[:-1])
    th = np.empty(nk)
    qv = np.empty(nk)
    z_inv = r["z_inv"] if r["z_inv"] is not None else r["zi"] + 80.0
    th_inv_base = r["th_ml"] + r["g_cloud"] * (z_inv - r["zi"])
    q_inv_base = r["q_top"]
    for k, z in enumerate(zm):
        if z <= r["zi"]:
            th[k] = r["th_ml"] + rng.normal(0.0, 0.03)
            qv[k] = r["q_ml"] * (1.0 - 0.05 * z / r["zi"])
        elif z <= z_inv:
            f = (z - r["zi"]) / max(z_inv - r["zi"], 1.0)
            th[k] = r["th_ml"] + r["g_cloud"] * (z - r["zi"])
            qv[k] = r["q_ml"] * 0.95 * (1 - f) + q_inv_base * f
        else:
            dz = z - z_inv
            jump = r["dth_inv"] * min(1.0, dz / 200.0)
            th[k] = th_inv_base + jump + r["g_free"] * dz
            if z > 12000.0:
                th[k] += 12e-3 * (z - 12000.0)
            qv[k] = max((q_inv_base - r["dq_inv"]) * np.exp(-dz / r["hq"]),
                        2.0e-6)
    th += 0.0  # keep float64
    # hydrostatic integration for interface pressures (virtual temperature)
    pw = np.empty(nk + 1)
    pw[0] = r["psfc"]
    for k in range(nk):
        exn_guess = (pw[k] / P0) ** RCP
        tv = th[k] * exn_guess * (1.0 + 0.608 * qv[k])
        pw[k + 1] = pw[k] * np.exp(-G * (zi_w[k + 1] - zi_w[k]) / (R_D * tv))
    pm = np.sqrt(pw[:-1] * pw[1:])
    exn = (pm / P0) ** RCP
    t = th * exn
    # cap supersaturation, then put some condensate where it is near-saturated
    qs = EP * esat_water(t) / (pm - (1 - EP) * esat_water(t))
    qv = np.minimum(qv, 1.02 * qs)
    qc = np.zeros(nk)
    qi = np.zeros(nk)
    if r.get("cloud"):
        rh = qv / qs
        cloudy = (rh > 0.9) & (zm > r["zi"]) & (zm < z_inv + 300)
        amt = rng.uniform(2e-6, 2e-4, size=nk) * cloudy
        frac_ice = np.clip((268.0 - t) / 30.0, 0.0, 1.0)
        qc = amt * (1 - frac_ice)
        qi = amt * frac_ice
    # thin high cirrus in some columns
    if rng.uniform() < 0.3:
        cirrus = (t < 235.0) & (t > 215.0)
        qi = qi + cirrus * rng.uniform(1e-7, 2e-5, size=nk)
    qnc = np.where(qc > 0, rng.uniform(3e7, 3e8, size=nk), 0.0)
    qni = np.where(qi > 0, rng.uniform(1e3, 3e5, size=nk), 0.0)
    wind = r["wind"]
    u = wind + r["shear"] * zm + rng.normal(0, 0.3, size=nk)
    v = 0.4 * wind + 0.5 * r["shear"] * zm + rng.normal(0, 0.3, size=nk)
    tke = np.empty(nk + 1)
    for k, z in enumerate(zi_w):
        if z <= r["zi"]:
            tke[k] = r["tke0"] * (0.4 + 0.6 * (1.0 - z / r["zi"]) ** 0.5)
        else:
            tke[k] = max(r["tke0"] * 0.4 * np.exp(-(z - r["zi"]) / 150.0),
                         0.01)
    tke[-1] = 0.0
    cf = np.clip((qc + qi) / 1e-4, 0.0, 1.0)
    return dict(zi_w=zi_w, zm=zm, th=th, qv=qv, qc=qc, qi=qi, qnc=qnc,
                qni=qni, u=u, v=v, pw=pw, pm=pm, exn=exn, t=t, tke=tke,
                cf=cf, pblh=r["zi"] + rng.uniform(-40.0, 40.0), ht=r["ht"])


def edge_columns(rng, nk, ztop):
    """Hand-built edge cases on top of perturbed regimes."""
    out = []
    base = lambda fam: sounding(rng, regime(rng, fam), nk, ztop)
    c = base("trade"); c["pblh"] = 1.0; out.append(("edge-pblh-below-first", c))
    c = base("continental"); c["pblh"] = 1.0e5; out.append(("edge-pblh-above-top", c))
    c = base("trade"); c["tke"][:] = 0.0; out.append(("edge-zero-tke", c))
    c = base("continental")
    c["qv"][:] = np.minimum(c["qv"], 1e-5); out.append(("edge-bone-dry", c))
    c = base("trade")
    es = esat_water(c["t"])
    qs = EP * es / (c["pm"] - (1 - EP) * es)
    low = c["zm"] < c["pblh"]
    c["qv"][low] = 1.05 * qs[low]; out.append(("edge-saturated-source", c))
    c = base("mixedphase")
    c["t"] -= 25.0; c["th"] = c["t"] / c["exn"]
    out.append(("edge-cold-column", c))
    c = base("overshoot"); c["tke"] *= 6.0; out.append(("edge-strong-tke", c))
    c = base("capped"); c["pblh"] = c["zm"][2]; out.append(("edge-pblh-at-level", c))
    return out


def to_f32(a):
    return np.asarray(a, dtype=np.float64).astype(np.float32)


def driver_case(cols, nk, nsteps, dt, has_qnc, has_qni, rng):
    """Arrays (k fastest, then column) for the driver harness."""
    ncol = len(cols)
    mass = {n: np.zeros((ncol, nk), np.float32) for n in DRIVER_MASS}
    full = {n: np.zeros((ncol, nk + 1), np.float32) for n in DRIVER_FULL}
    sfc = {n: np.zeros(ncol, np.float32) for n in DRIVER_SFC}
    for i, (_, c) in enumerate(cols):
        ht = c["ht"]
        mass["u"][i] = to_f32(c["u"]); mass["v"][i] = to_f32(c["v"])
        mass["t"][i] = to_f32(c["t"]); mass["qv"][i] = to_f32(c["qv"])
        mass["qc"][i] = to_f32(c["qc"]); mass["qi"][i] = to_f32(c["qi"])
        mass["qnc"][i] = to_f32(c["qnc"]); mass["qni"][i] = to_f32(c["qni"])
        mass["p"][i] = to_f32(c["pm"]); mass["z"][i] = to_f32(c["zm"] + ht)
        mass["pi"][i] = to_f32(c["exn"])
        mass["cldfra"][i] = to_f32(c["cf"])
        mass["cldfra_old"][i] = to_f32(0.5 * c["cf"])
        # the deep part a deep CAM scheme would have left (zero, or a
        # random positive amount on some columns)
        if rng.uniform() < 0.25:
            mass["dlf"][i] = to_f32(rng.uniform(0, 1e-7, size=nk))
            full["cmfmc"][i] = to_f32(rng.uniform(0, 0.02, size=nk + 1))
            sfc["rliq"][i] = np.float32(rng.uniform(0, 1e-8))
        full["p8w"][i] = to_f32(c["pw"])
        full["z_at_w"][i] = to_f32(c["zi_w"] + ht)
        full["tke_pbl"][i] = to_f32(c["tke"])
        sfc["pblh"][i] = np.float32(c["pblh"]); sfc["ht"][i] = np.float32(ht)
        # WRF's Registry starts CUSH at zero; some columns start from a
        # carried value (restart-like) or the scheme's own exit value -1.
        u = rng.uniform()
        sfc["cush"][i] = np.float32(0.0 if u < 0.5 else
                                    (-1.0 if u < 0.7 else rng.uniform(500, 4000)))
        # HTOP/HBOT as WRF's phy_init leaves them on a cold start
        sfc["cutop"][i] = np.float32(1.0)
        sfc["cubot"][i] = np.float32(nk)
    head = np.array([DRIVER_MAGIC, 1, ncol, nk, nsteps, int(has_qnc),
                     int(has_qni)], dtype=np.int32)
    blob = [head.tobytes(), np.float32(dt).tobytes()]
    for n in DRIVER_MASS:
        blob.append(np.ascontiguousarray(mass[n]).tobytes())
    for n in DRIVER_FULL:
        blob.append(np.ascontiguousarray(full[n]).tobytes())
    for n in DRIVER_SFC:
        blob.append(sfc[n].tobytes())
    return b"".join(blob)


def core_case(cols, nk, dt, rng):
    """r8 compute_uwshcu_inv inputs: the driver's own mapping in float64
    plus an r8 jitter no float32 value carries, CAM orientation."""
    ncol = len(cols)
    cpair = 1004.64
    grav = 9.80616
    mass = {n: np.zeros((ncol, nk)) for n in CORE_MASS}
    full = {n: np.zeros((ncol, nk + 1)) for n in CORE_FULL}
    trac = np.zeros((ncol, 5, nk))
    sfc = {n: np.zeros(ncol) for n in CORE_SFC}
    for i, (_, c) in enumerate(cols):
        jit = lambda n: 1.0 + rng.uniform(-1e-9, 1e-9, size=n)
        qv = c["qv"]
        qh = np.maximum(qv / (1 + qv), 1e-30) * jit(nk)
        pdel = (c["pw"][:-1] - c["pw"][1:])
        flip = slice(None, None, -1)
        mass["p0"][i] = c["pm"][flip] * jit(nk)
        mass["z0"][i] = c["zm"][flip] * jit(nk)
        mass["dp0"][i] = pdel[flip]
        mass["dpdry0"][i] = (pdel * (1 - qh))[flip]
        mass["u0"][i] = c["u"][flip] * jit(nk)
        mass["v0"][i] = c["v"][flip] * jit(nk)
        mass["qv0"][i] = qh[flip]
        mass["ql0"][i] = (c["qc"] / (1 + qv))[flip]
        mass["qi0"][i] = (c["qi"] / (1 + qv))[flip]
        mass["t0"][i] = c["t"][flip] * jit(nk)
        mass["s0"][i] = (cpair * c["t"] + grav * c["zm"])[flip]
        mass["cldfrct"][i] = c["cf"][flip]
        mass["concldfrct"][i] = 0.5 * c["cf"][flip]
        full["ps0"][i] = c["pw"][flip]
        full["zs0"][i] = c["zi_w"][flip]
        full["tke"][i] = c["tke"][flip]
        trac[i, 0] = qh[flip]
        trac[i, 1] = mass["ql0"][i]
        trac[i, 2] = mass["qi0"][i]
        trac[i, 3] = (c["qnc"] / (1 + qv))[flip]
        trac[i, 4] = (c["qni"] / (1 + qv))[flip]
        sfc["pblh"][i] = c["pblh"]
        sfc["cush"][i] = 0.0 if rng.uniform() < 0.6 else rng.uniform(500, 4000)
    head = np.array([CORE_MAGIC, 1, ncol, nk], dtype=np.int32)
    blob = [head.tobytes()]
    for n in CORE_MASS:
        blob.append(np.ascontiguousarray(mass[n]).tobytes())
    for n in CORE_FULL:
        blob.append(np.ascontiguousarray(full[n]).tobytes())
    blob.append(np.ascontiguousarray(trac).tobytes())
    for n in CORE_SFC:
        blob.append(sfc[n].tobytes())
    blob.append(np.float64(dt).tobytes())
    return b"".join(blob)


GRIDS = (
    # name, nk, ztop, dt, nsteps, has_qnc, has_qni
    ("g40", 40, 20000.0, 30.0, 4, True, True),
    ("g56", 56, 22000.0, 12.0, 3, False, True),
    ("g33", 33, 18000.0, 60.0, 3, False, False),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261008)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for gi, (name, nk, ztop, dt, nsteps, has_qnc, has_qni) in enumerate(GRIDS):
        rng = np.random.default_rng(a.seed + 7919 * gi)
        per = a.per_family if gi == 0 else max(3, a.per_family // 3)
        cols = []
        for fam in FAMILIES:
            for j in range(per):
                cols.append((f"{fam}-{j}", sounding(rng, regime(rng, fam),
                                                     nk, ztop)))
        cols += edge_columns(rng, nk, ztop)
        (out / f"cases-{name}.bin").write_bytes(
            driver_case(cols, nk, nsteps, dt, has_qnc, has_qni, rng))
        (out / f"core-{name}.bin").write_bytes(core_case(cols, nk, dt, rng))
        index[name] = dict(nk=nk, ztop=ztop, dt=dt, nsteps=nsteps,
                           has_qnc=has_qnc, has_qni=has_qni,
                           columns=[c[0] for c in cols])
    (out / "cases-index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(json.dumps({k: len(v["columns"]) for k, v in index.items()}))


if __name__ == "__main__":
    main()
