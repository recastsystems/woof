"""Write the slab column oracle's input cases (deterministic).

Every case is one surface column that WRF's SLAB is run on for ``NSTEP``
consecutive surface calls with its forcing held, so the oracle grades the
first call and the state the scheme carries from call to call.  The regimes
are named (``label``) so a mismatch reads as a place in the scheme, not as a
row number:

* day / night (GSW from 0 to 950 W m-2, radiation on and off);
* wet / dry (MAVAIL and FLQC from 0 to the water value; FLQC = 0 exercises
  WRF's QSFC guard);
* snow (SNOWC = 1 with IFSNOW 0, 1 and 2, skins below, at and above the
  273.16 K cap, and warming columns that cross it during the run);
* land / water / ice edges (XLAND 1 and 2 side by side, and the sea-ice
  columns real.exe makes from water: XLAND = 1, the ice row's THERIN and
  TMN = 271.4 K);
* sea-ice threshold crossings (a former water column converted to ice whose
  skin warms through 273.16 K under snow);
* soil sub-step counts NSOIL 1, 2, 4, 5 and 8 (DELTSM 1 s to 300 s);
* surface pressure from 60 kPa (high terrain) to 104 kPa.

The soil geometry is init_soil_depth_1 in float32, as real.exe builds it.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

f32 = np.float32
NSL = 5
NSTEP = 36
MAGIC_IN = 5150

COLUMN_FIELDS = ("t1", "qv1", "p1", "flhc", "flqc", "psfc", "xland", "tmn",
                 "hfx", "qfx", "lh", "tsk", "qsfc", "chklowq", "gsw", "glw",
                 "capg", "thc", "snowc", "emiss", "mavail")


def soil_depths():
    """init_soil_depth_1 (module_soil_pre.F:1078-1126) in float32."""
    dzs = np.zeros(NSL, f32)
    zs = np.zeros(NSL, f32)
    dzs[0] = f32(0.01)
    zs[0] = f32(0.5) * dzs[0]
    for layer in range(1, NSL):
        dzs[layer] = f32(2) * dzs[layer - 1]
        zs[layer] = (zs[layer - 1] + f32(0.5) * dzs[layer - 1]
                     + f32(0.5) * dzs[layer])
    return zs, dzs


def initial_profile(tsk, tmn, zs):
    """init_soil_1_real's land profile (module_soil_pre.F:1287-1340)."""
    tsk = f32(tsk)
    tmn = f32(tmn)
    out = np.zeros(NSL, f32)
    for layer in range(NSL):
        out[layer] = ((tsk * (zs[-1] - zs[layer])
                       + tmn * (zs[layer] - zs[0])) / (zs[-1] - zs[0]))
    return out


def qsat(t, p_pa):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p_pa - es)


def make_cases(seed: int = 461):
    rng = np.random.default_rng(seed)
    zs, dzs = soil_depths()
    dts = (1.0, 6.0, 30.0, 60.0, 72.0, 120.0, 180.0, 300.0)
    cases = []

    def base(**over):
        c = dict(label="", ifsnow=1, radiation=1, deltsm=60.0,
                 t1=290.0, qv1=0.010, p1=99000.0, flhc=20.0, flqc=0.004,
                 psfc=100000.0, xland=1.0, tmn=288.0, hfx=50.0, qfx=5e-5,
                 lh=125.0, tsk=292.0, qsfc=0.012, chklowq=0.3, gsw=500.0,
                 glw=350.0, capg=0.0, thc=0.04, snowc=0.0, emiss=0.95,
                 mavail=0.30, tslb=None, perturb=0.0)
        c.update(over)
        return c

    # --- targeted regimes ------------------------------------------------
    for dt in dts:
        cases.append(base(label=f"day-moist dt={dt:g}", deltsm=dt,
                          gsw=800.0, tsk=300.0, t1=296.0, qv1=0.014,
                          mavail=0.5, flqc=0.006))
        cases.append(base(label=f"night-dry dt={dt:g}", deltsm=dt,
                          gsw=0.0, glw=260.0, tsk=281.0, t1=284.0,
                          qv1=0.004, mavail=0.05, flqc=0.0004, flhc=6.0))
    for ifs in (0, 1, 2):
        for tsk0, gsw in ((268.0, 900.0), (272.9, 700.0), (273.16, 400.0),
                          (275.0, 0.0), (279.0, 600.0)):
            cases.append(base(label=f"snow ifsnow={ifs} tsk={tsk0:g} "
                                    f"gsw={gsw:g}",
                              ifsnow=ifs, snowc=1.0, tsk=tsk0, t1=tsk0 + 1.5,
                              tmn=271.0, qv1=0.003, gsw=gsw, glw=300.0,
                              mavail=0.95, thc=0.05, flqc=0.003, deltsm=120.0))
    # Sea ice made by real.exe from cold water (adjust_for_seaice_pre):
    # XLAND=1, the ice row (THERIN 5 -> THC 0.05), TMN=271.4 K.
    for tsk0, gsw, snow in ((258.0, 0.0, 1.0), (271.0, 650.0, 1.0),
                            (272.8, 900.0, 1.0), (265.0, 300.0, 0.0),
                            (273.5, 850.0, 1.0), (240.0, 0.0, 0.0)):
        cases.append(base(label=f"sea-ice tsk={tsk0:g} gsw={gsw:g} "
                                f"snow={snow:g}",
                          xland=1.0, thc=0.05, tmn=271.4, tsk=tsk0,
                          t1=tsk0 - 2.0, qv1=0.0015, mavail=0.95,
                          snowc=snow, gsw=gsw, glw=220.0, flqc=0.002,
                          deltsm=180.0, emiss=0.95))
    # Open water beside land: untouched TSK/HFX/QFX/LH, QSFC rebuilt.
    for flqc in (0.0, 0.005, 0.02):
        cases.append(base(label=f"water flqc={flqc:g}", xland=2.0,
                          thc=0.06, flqc=flqc, mavail=1.0, tsk=295.0,
                          hfx=12.5, qfx=8.0e-5, lh=200.0))
    cases.append(base(label="water xland=1.5 edge", xland=1.5, thc=0.06))
    cases.append(base(label="land xland=1.4999 edge",
                      xland=float(np.nextafter(f32(1.5), f32(0.0))),
                      thc=0.03))
    # Land with FLQC exactly zero (WRF's QSFC guard) and radiation off.
    cases.append(base(label="land flqc=0", flqc=0.0))
    cases.append(base(label="land radiation off", radiation=0, gsw=0.0,
                      glw=0.0))
    cases.append(base(label="land radiation off daytime", radiation=0))
    # High terrain and extreme skins.
    cases.append(base(label="high terrain", psfc=60000.0, p1=59300.0,
                      t1=268.0, tsk=275.0, qv1=0.002, tmn=270.0))
    cases.append(base(label="hot desert", tsk=335.0, t1=312.0, qv1=0.006,
                      mavail=0.02, gsw=950.0, thc=0.02, flhc=45.0,
                      flqc=0.0003, deltsm=300.0))
    cases.append(base(label="cold polar night", tsk=232.0, t1=236.0,
                      qv1=0.0002, gsw=0.0, glw=160.0, tmn=245.0, thc=0.02))
    cases.append(base(label="saturated swamp", mavail=1.0, flqc=0.02,
                      qv1=0.019, t1=300.0, tsk=302.0, thc=0.055))

    # --- randomized fill over the admitted envelope ----------------------
    while len(cases) < 96:
        psfc = float(rng.uniform(62000.0, 104000.0))
        t1 = float(rng.uniform(235.0, 312.0))
        tsk0 = float(np.clip(t1 + rng.uniform(-14.0, 22.0), 225.0, 340.0))
        land = rng.random() < 0.8
        snow = float(rng.random() < 0.25 and tsk0 < 280.0)
        p1 = psfc - float(rng.uniform(300.0, 1500.0))
        qv1 = float(min(rng.uniform(0.05, 0.95) * qsat(t1, p1), 0.025))
        mav = float(rng.uniform(0.02, 1.0))
        cases.append(base(
            label=f"random {len(cases)}",
            ifsnow=int(rng.integers(0, 3)), radiation=int(rng.random() < 0.9),
            deltsm=float(rng.choice(dts)), t1=t1, qv1=qv1, p1=p1,
            flhc=float(rng.uniform(0.5, 60.0)),
            flqc=float(0.0 if rng.random() < 0.05
                       else rng.uniform(1e-5, 0.025) * mav),
            psfc=psfc, xland=1.0 if land else 2.0,
            tmn=float(tsk0 + rng.uniform(-12.0, 12.0)),
            hfx=float(rng.uniform(-80.0, 400.0)),
            qfx=float(rng.uniform(-2e-5, 3e-4)),
            lh=float(rng.uniform(-50.0, 700.0)), tsk=tsk0,
            qsfc=float(rng.uniform(1e-4, 0.03)),
            chklowq=float(rng.uniform(0.0, 1.0)),
            gsw=float(rng.uniform(0.0, 950.0) if rng.random() < 0.6 else 0.0),
            glw=float(rng.uniform(150.0, 450.0)),
            capg=float(rng.uniform(0.0, 3e6)),
            thc=float(rng.uniform(0.02, 0.06)), snowc=snow,
            emiss=float(rng.uniform(0.85, 0.99)), mavail=mav,
            perturb=float(rng.uniform(0.0, 3.0))))

    out = []
    for c in cases:
        tslb = (initial_profile(c["tsk"], c["tmn"], zs)
                if c["tslb"] is None else np.asarray(c["tslb"], f32))
        if c["perturb"]:
            tslb = (tslb + rng.uniform(-c["perturb"], c["perturb"], NSL)
                    ).astype(f32)
        c = dict(c)
        c["tslb"] = [float(v) for v in tslb.astype(f32)]
        c.pop("perturb")
        out.append(c)
    return out, zs, dzs


def write_input(path: Path, cases, zs, dzs, nstep: int = NSTEP) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<ii", MAGIC_IN, len(cases)))
        for c in cases:
            fh.write(struct.pack("<iii", int(c["ifsnow"]), int(c["radiation"]),
                                 int(nstep)))
            fh.write(np.asarray([c["deltsm"]], f32).tobytes())
            fh.write(np.asarray([c[k] for k in COLUMN_FIELDS], f32).tobytes())
            fh.write(np.asarray(c["tslb"], f32).tobytes())
            fh.write(np.asarray(zs, f32).tobytes())
            fh.write(np.asarray(dzs, f32).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--nstep", type=int, default=NSTEP)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases, zs, dzs = make_cases()
    write_input(args.out_dir / "slab-input.bin", cases, zs, dzs, args.nstep)
    (args.out_dir / "slab-cases.json").write_text(json.dumps(
        {"nstep": args.nstep, "zs": [float(v) for v in zs],
         "dzs": [float(v) for v in dzs], "fields": list(COLUMN_FIELDS),
         "cases": cases}, indent=1), encoding="utf-8")
    print(f"{len(cases)} cases x {args.nstep} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
