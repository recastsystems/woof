#!/usr/bin/env python3
"""Generate the MSKF (cu_physics = 11) column-oracle inputs.

Writes, into the directory named on the command line,

  cases_i32.bin   ncase x NI little-endian int32 control words
  cases_f32.bin   ncase records of little-endian float32 (layout below)
  cases.json      the case table, the field order and the layout

Every float is generated in float64 and rounded ONCE to float32 here, so the
oracle and the port read the very same words.  The case table is built to
reach branches, not to be meteorologically typical: deep, shallow,
no-trigger, the "double counting" saturated environment, cold and warm
phase routes, every phase-feedback arm of the driver, the two dx thresholds
(24.999 km for the scale factor and 25 km for the trigger), an NCA-held
column, an adaptive-step column, and the aercu_opt 0/1/2 arms with clean and
polluted aerosol.

Layout of one float32 record (nz mass levels, nz+1 full levels):
  scalars  NF words: see SCALARS
  levels   len(LEVEL_FIELDS) x nz words, field-major
  w        nz+1 words (full levels)
  aerocu   10 x nz words, species-major (WRF's aerocu(i,k,j,s))
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

NZ = 40
R_D = 287.0
CP = 1004.5
G = 9.81
P_TOP = 5000.0

INTS = ("warm_rain", "f_qr", "f_qi", "f_qs", "aercu_opt", "kf_edrates",
        "adapt_step", "stepcu", "trigger")
SCALARS = ("dx", "dt", "cudt", "aercu_fct", "nca", "hfx", "ust", "pblh",
           "zol", "xland")
LEVEL_FIELDS = ("u", "v", "t", "th", "qv", "p", "rho", "pi", "dz8w",
                "w0avg", "rqvften")

# Phase arms of the MSKF feedback block (module_cu_mskf.F:6646-6685) and of
# the driver's F_QR switch (:3956-3967), as (warm_rain, F_QR, F_QI, F_QS).
PHASES = {
    "ice_snow": (0, 1, 1, 1),     # Thompson/Morrison/WSM6 shape
    "warm": (1, 1, 0, 0),         # Kessler
    "nosnow": (0, 1, 0, 0),       # WSM3 shape: melting-level closure
    "snow_only": (0, 1, 0, 1),    # F_QS without F_QI: ice folded into snow
    "no_rain": (0, 0, 0, 0),      # Eta-shape: rain folded into cloud
}


def sounding(*, ps, ts, rh_bl, rh_mid, rh_top, z_pbl, cap_dt=0.0,
             z_cap=None, trop_p=15000.0, lapse=6.5e-3, u0=2.0, du=1.5e-3,
             v0=1.0, dv=0.5e-3, w_amp=0.2, w0_amp=0.15, w_top=6000.0,
             rqvften_amp=1.0e-7):
    """One column on NZ mass levels, built in float64."""
    s = np.arange(NZ + 1, dtype=np.float64) / NZ
    eta = 1.0 - s ** 1.35
    p8w = P_TOP + eta * (ps - P_TOP)
    p8w[-1] = P_TOP
    p = 0.5 * (p8w[:-1] + p8w[1:])
    # Temperature as a function of pressure: dry adiabatic to the PBL top,
    # then a constant lapse rate in z (a power law in p), an optional
    # capping inversion, isothermal above the tropopause.
    p_pbl = ps * (1.0 - G * z_pbl / (CP * ts)) ** (CP / R_D)
    t_pbl = ts * (p_pbl / ps) ** (R_D / CP)
    t = np.where(p >= p_pbl, ts * (p / ps) ** (R_D / CP),
                 t_pbl * (p / p_pbl) ** (R_D * lapse / G))
    t_trop = t_pbl * (trop_p / p_pbl) ** (R_D * lapse / G)
    t = np.where(p < trop_p, t_trop, t)
    # Heights and the inversion need z; integrate hydrostatically.
    def heights(tt, qq):
        tv = tt * (1.0 + 0.608 * qq)
        dz = R_D * tv / G * np.log(p8w[:-1] / p8w[1:])
        z8w = np.concatenate([[0.0], np.cumsum(dz)])
        return dz, 0.5 * (z8w[:-1] + z8w[1:]), z8w
    q = np.zeros(NZ)
    for _ in range(3):
        dz, z, z8w = heights(t, q)
        if z_cap is not None and cap_dt:
            t_cap = t + cap_dt / (1.0 + np.exp(-(z - z_cap) / 120.0))
        else:
            t_cap = t
        rh = np.where(z <= z_pbl, rh_bl,
                      np.where(p >= 40000.0, rh_mid, rh_top))
        es = 611.2 * np.exp(17.67 * (t_cap - 273.15) / (t_cap - 29.65))
        qs = 0.622 * es / (p - es)
        q = np.maximum(rh * qs, 1.0e-6)
    t = t_cap
    dz, z, z8w = heights(t, q)
    u = u0 + du * z
    v = v0 + dv * z
    w = w_amp * np.sin(np.pi * np.clip(z8w / w_top, 0.0, 1.0))
    w0avg = w0_amp * np.sin(np.pi * np.clip(z / w_top, 0.0, 1.0))
    rho = p / (R_D * t * (1.0 + 0.608 * q))
    pi = (p / 1.0e5) ** (R_D / CP)
    th = t / pi
    rqvften = rqvften_amp * np.cos(np.pi * z / 12000.0)
    return {"u": u, "v": v, "t": t, "th": th, "qv": q, "p": p, "rho": rho,
            "pi": pi, "dz8w": dz, "w0avg": w0avg, "rqvften": rqvften,
            "w": w}


SOUNDINGS = {
    # deep convection, maritime tropical
    "tropical_deep": dict(ps=101000.0, ts=301.0, rh_bl=0.90, rh_mid=0.75,
                          rh_top=0.5, z_pbl=600.0, w_amp=0.25, w0_amp=0.2),
    # deep convection over land: hot, dry mid levels (downdraft)
    "continental_deep": dict(ps=96000.0, ts=306.0, rh_bl=0.62, rh_mid=0.35,
                             rh_top=0.3, z_pbl=1800.0, w_amp=0.4,
                             w0_amp=0.35, du=4.0e-3),
    # strong capping inversion at ~1.5 km
    "capped": dict(ps=100000.0, ts=303.0, rh_bl=0.8, rh_mid=0.4, rh_top=0.3,
                   z_pbl=1200.0, cap_dt=6.0, z_cap=1500.0, w0_amp=0.05),
    # trade cumulus under a low inversion (shallow regime)
    "shallow_trade": dict(ps=101500.0, ts=299.0, rh_bl=0.85, rh_mid=0.25,
                          rh_top=0.2, z_pbl=500.0, cap_dt=4.0, z_cap=1600.0,
                          w0_amp=0.08, w_top=2500.0, trop_p=16000.0),
    # marginal, weakly forced
    "marginal": dict(ps=100500.0, ts=297.0, rh_bl=0.75, rh_mid=0.55,
                     rh_top=0.4, z_pbl=900.0, w_amp=0.02, w0_amp=0.01),
    # cold stable column: never triggers
    "stable_cold": dict(ps=102000.0, ts=268.0, rh_bl=0.6, rh_mid=0.5,
                        rh_top=0.4, z_pbl=200.0, lapse=4.0e-3, w0_amp=0.02),
    # dry mid levels, strong evaporative downdraft
    "dry_midlevel": dict(ps=99000.0, ts=304.0, rh_bl=0.75, rh_mid=0.18,
                         rh_top=0.15, z_pbl=1400.0, w0_amp=0.3),
    # high terrain
    "high_terrain": dict(ps=75000.0, ts=292.0, rh_bl=0.7, rh_mid=0.5,
                         rh_top=0.35, z_pbl=1500.0, w0_amp=0.25),
    # near-saturated environment: the double-counting refusal (envRHavg)
    "saturated": dict(ps=100500.0, ts=298.0, rh_bl=0.995, rh_mid=0.995,
                      rh_top=0.995, z_pbl=400.0, w0_amp=0.3),
    # cold season: low freezing level, ice processes in the updraft
    "cold_season": dict(ps=100000.0, ts=280.0, rh_bl=0.85, rh_mid=0.7,
                        rh_top=0.5, z_pbl=700.0, w0_amp=0.3,
                        trop_p=22000.0),
    # strong directional and speed shear (momentum transport)
    "sheared": dict(ps=100500.0, ts=302.0, rh_bl=0.85, rh_mid=0.6,
                    rh_top=0.4, z_pbl=900.0, u0=-5.0, du=5.0e-3, v0=8.0,
                    dv=-2.0e-3, w0_amp=0.3),
    # subsiding column: negative running-mean w, no trigger
    "subsiding": dict(ps=101000.0, ts=301.0, rh_bl=0.9, rh_mid=0.7,
                      rh_top=0.5, z_pbl=600.0, w_amp=-0.2, w0_amp=-0.15),
    # very unstable, very moist: tall cloud, strong mass flux
    "explosive": dict(ps=100800.0, ts=304.0, rh_bl=0.92, rh_mid=0.8,
                      rh_top=0.6, z_pbl=800.0, w_amp=0.6, w0_amp=0.5,
                      lapse=7.2e-3, trop_p=12000.0),
    # nocturnal: stable surface layer (negative HFX, positive z/L)
    "nocturnal": dict(ps=100000.0, ts=296.0, rh_bl=0.85, rh_mid=0.65,
                      rh_top=0.4, z_pbl=300.0, w0_amp=0.2),
}

SURFACE = {
    "nocturnal": dict(hfx=-25.0, ust=0.15, pblh=250.0, zol=0.8, xland=1.0),
    "shallow_trade": dict(hfx=20.0, ust=0.3, pblh=600.0, zol=-0.2, xland=2.0),
    "tropical_deep": dict(hfx=40.0, ust=0.35, pblh=700.0, zol=-0.4,
                          xland=2.0),
    "explosive": dict(hfx=350.0, ust=0.6, pblh=1500.0, zol=-3.0, xland=1.0),
    "continental_deep": dict(hfx=420.0, ust=0.5, pblh=2200.0, zol=-6.0,
                             xland=1.0),
}
SURFACE_DEFAULT = dict(hfx=150.0, ust=0.4, pblh=1100.0, zol=-1.0, xland=1.0)

# CESM-like aerosol profiles in ug m-3 for the ten aerocu species
# (Registry.EM_COMMON:1289-1298 order: dust1-4, seasalt, sulfate,
# phobcar, phibcar, phoocar, phiocar).
AEROSOL_CLEAN = np.array([0.4, 0.8, 0.6, 0.2, 2.0, 0.6, 0.02, 0.05, 0.08,
                          0.2])
AEROSOL_POLLUTED = np.array([8.0, 15.0, 9.0, 3.0, 1.0, 14.0, 0.6, 1.2, 2.5,
                             6.0])


def aerosol_profile(base, p):
    """Exponential decay with a 2.5 km-ish pressure scale."""
    scale = (p / p[0]) ** 2.2
    return base[:, None] * scale[None, :]


def build_cases():
    cases = []
    dxs = (1000.0, 3000.0, 9000.0, 27000.0)
    for name in SOUNDINGS:
        for dx in dxs:
            cases.append(dict(sounding=name, dx=dx, phase="ice_snow",
                              aercu_opt=0, aerosol="none", dt=dx / 200.0
                              if dx >= 3000.0 else 6.0, cudt=5.0,
                              nca=-100.0, adapt_step=0))
    # the two scale thresholds, on both sides
    for dx in (24998.0, 24999.0, 24999.5, 25000.0):
        cases.append(dict(sounding="tropical_deep", dx=dx, phase="ice_snow",
                          aercu_opt=0, aerosol="none", dt=60.0, cudt=5.0,
                          nca=-100.0, adapt_step=0))
    # every feedback phase arm on a deep and a cold column
    for phase in PHASES:
        if phase == "ice_snow":
            continue
        for snd, dx in (("continental_deep", 3000.0), ("cold_season", 3000.0),
                        ("shallow_trade", 3000.0), ("explosive", 3000.0),
                        ("tropical_deep", 9000.0), ("sheared", 27000.0)):
            cases.append(dict(sounding=snd, dx=dx, phase=phase,
                              aercu_opt=0, aerosol="none", dt=15.0,
                              cudt=5.0, nca=-100.0, adapt_step=0))
    # NCA hold: an active hold skips the column entirely
    cases.append(dict(sounding="tropical_deep", dx=9000.0, phase="ice_snow",
                      aercu_opt=0, aerosol="none", dt=45.0, cudt=5.0,
                      nca=900.0, adapt_step=0))
    # the hold boundary NCA = 0.5*DT is still held
    cases.append(dict(sounding="tropical_deep", dx=9000.0, phase="ice_snow",
                      aercu_opt=0, aerosol="none", dt=45.0, cudt=5.0,
                      nca=22.5, adapt_step=0))
    # cudt = 0 (every step) and the adaptive-step W0AVG weights
    cases.append(dict(sounding="explosive", dx=3000.0, phase="ice_snow",
                      aercu_opt=0, aerosol="none", dt=15.0, cudt=0.0,
                      nca=-100.0, adapt_step=0))
    cases.append(dict(sounding="explosive", dx=3000.0, phase="ice_snow",
                      aercu_opt=0, aerosol="none", dt=15.0, cudt=5.0,
                      nca=-100.0, adapt_step=1))
    # aerosol-aware updraft microphysics, aercu_opt 1 and 2, clean/polluted
    for snd in ("tropical_deep", "continental_deep", "cold_season",
                "explosive", "shallow_trade", "sheared"):
        for opt, aero in ((1, "clean"), (2, "polluted"), (1, "polluted"),
                          (2, "clean")):
            for dx in (3000.0, 27000.0):
                cases.append(dict(sounding=snd, dx=dx, phase="ice_snow",
                                  aercu_opt=opt, aerosol=aero,
                                  dt=15.0 if dx < 9000.0 else 120.0,
                                  cudt=5.0, nca=-100.0, adapt_step=0))
    # aerosol path on more deep, raining columns and on the other phase arms
    for snd, dx in (("tropical_deep", 1000.0), ("tropical_deep", 9000.0),
                    ("high_terrain", 9000.0), ("sheared", 9000.0),
                    ("continental_deep", 9000.0), ("dry_midlevel", 27000.0),
                    ("explosive", 1000.0)):
        for opt, aero in ((1, "clean"), (2, "polluted")):
            cases.append(dict(sounding=snd, dx=dx, phase="ice_snow",
                              aercu_opt=opt, aerosol=aero, dt=6.0
                              if dx < 3000.0 else dx / 200.0, cudt=5.0,
                              nca=-100.0, adapt_step=0))
    for phase in ("warm", "nosnow", "snow_only", "no_rain"):
        cases.append(dict(sounding="tropical_deep", dx=3000.0, phase=phase,
                          aercu_opt=1, aerosol="polluted", dt=15.0,
                          cudt=5.0, nca=-100.0, adapt_step=0))
    return cases


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    ints = []
    floats = []
    for case in cases:
        col = sounding(**SOUNDINGS[case["sounding"]])
        sfc = dict(SURFACE_DEFAULT)
        sfc.update(SURFACE.get(case["sounding"], {}))
        warm, fqr, fqi, fqs = PHASES[case["phase"]]
        dt = float(np.float32(case["dt"]))
        cudt = case["cudt"]
        stepcu = 1 if cudt <= 0.0 else max(int(math.floor(
            cudt * 60.0 / dt + 0.5)), 1)
        ints.append([warm, fqr, fqi, fqs, case["aercu_opt"], 1,
                     case["adapt_step"], stepcu, 1])
        scal = [case["dx"], dt, cudt, 1.0, case["nca"], sfc["hfx"],
                sfc["ust"], sfc["pblh"], sfc["zol"], sfc["xland"]]
        rec = [np.asarray(scal, dtype=np.float32)]
        for name in LEVEL_FIELDS:
            rec.append(np.asarray(col[name], dtype=np.float32))
        rec.append(np.asarray(col["w"], dtype=np.float32))
        if case["aerosol"] == "clean":
            aer = aerosol_profile(AEROSOL_CLEAN, col["p"])
        elif case["aerosol"] == "polluted":
            aer = aerosol_profile(AEROSOL_POLLUTED, col["p"])
        else:
            aer = np.zeros((10, NZ))
        rec.append(np.asarray(aer, dtype=np.float32).reshape(-1))
        floats.append(np.concatenate(rec))
    ints = np.asarray(ints, dtype="<i4")
    floats = np.asarray(floats, dtype="<f4")
    (out_dir / "cases_i32.bin").write_bytes(ints.tobytes())
    (out_dir / "cases_f32.bin").write_bytes(floats.tobytes())
    meta = {"nz": NZ, "ncase": len(cases), "ints": list(INTS),
            "scalars": list(SCALARS), "level_fields": list(LEVEL_FIELDS),
            "record_floats": int(floats.shape[1]), "cases": cases}
    (out_dir / "cases.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(cases)} cases, {NZ} levels, "
          f"{floats.shape[1]} floats per record")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
