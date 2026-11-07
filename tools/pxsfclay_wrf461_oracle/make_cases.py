"""Build the PX surface-layer oracle's input files.

Every number here is an INPUT: a column state the oracle hands both WRF's
Fortran and the CUDA kernel.  Nothing is an expected value.

Nine regime families of eight columns each (72 columns), chosen to reach
every branch of PXSFCLAY1D and of the sea-ice wrapper:

  unstable-day     land, skin warmer than the air, upward fluxes, deep PBL
  stable-night     land, skin colder than the air, shallow PBL, light wind
  neutral-windy    land, skin potential temperature at the air's, 10-25 m/s
  water            XLAND = 2: Charnock roughness, unaveraged u*
  cold-land        skin below 273.15 K: the ice-saturation branch of E1
  high-roughness   z0 of 0.8-2.5 m under a 30-50 m first half level
  calm-convective  zero or near-zero wind lifted by the convective velocity
  edge             very stable (BR >= RICUT), |TSTV| < 1e-5, 0 < L < 2,
                   MAVAIL = 0, a skin exactly at 273.15 K, the -250 W m-2
                   floor
  sea-ice          fractional ice from 0.02 to 1, blended skin, cold SST,
                   SST above 273 K on the first three steps

Five configurations (files), each several steps long:

  plain-dx12k      PXSFCLAY, dx 12 km (subgrid velocity on), steps 1-6
  plain-dx3k-lsm   PXSFCLAY, dx 3 km, a land model rewriting QSFC/HFX/QFX/ZNT
                   between calls on land columns, steps 1-6
  plain-restart    PXSFCLAY from a warm state at ITIMESTEP 37 (QSFC carried
                   from the land model, some columns QSFC <= 0), 4 steps
  ice-002          PXSFCLAY_SEAICE_WRAPPER, XICE_THRESHOLD 0.02, steps 1-6
  ice-05-tice      the wrapper, threshold 0.5, TICE2TSK_IF2COLD true, land
                   model overrides, steps 1-5

usage: make_cases.py OUT_DIR
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

F = np.float32
MAGIC_IN = 0x50585346
STATE = ("regime", "hfx", "qfx", "lh", "mol", "rmol", "qsfc", "gz1oz0",
         "wspd", "br", "psim", "psih", "znt", "zol", "ust", "cpm", "chs2",
         "cqs2", "chs", "flhc", "flqc", "qgh", "sst")
FORCING = ("u", "v", "t", "th", "qv", "p", "dz8w", "psfc", "tsk", "pblh",
           "mavail", "xland", "xice")
OVERRIDE = ("flag", "qsfc", "hfx", "qfx", "znt")
FAMILIES = ("unstable-day", "stable-night", "neutral-windy", "water",
            "cold-land", "high-roughness", "calm-convective", "edge",
            "sea-ice")
PER_FAMILY = 8
RCP = F(287.0) / F(F(7.0) * F(287.0) / F(2.0))


def _qsat(t, p):
    """Bolton-style saturation mixing ratio, only to build plausible inputs."""
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def _column(rng, family: str, n: int) -> dict:
    """One column's static forcing and initial state."""
    c = {}
    psfc = rng.uniform(85000.0, 102500.0)
    dz = rng.uniform(40.0, 70.0)
    t = rng.uniform(280.0, 305.0)
    rh = rng.uniform(0.3, 0.9)
    wind = rng.uniform(2.0, 8.0)
    znt = rng.uniform(0.03, 0.5)
    xland, xice, mavail = 1.0, 0.0, rng.uniform(0.2, 0.9)
    pblh = rng.uniform(300.0, 1500.0)
    hfx0, qfx0 = rng.uniform(-30.0, 150.0), rng.uniform(0.0, 1.0e-4)
    dtsk = rng.uniform(-2.0, 4.0)
    sst = 290.0
    if family == "unstable-day":
        dtsk = rng.uniform(3.0, 14.0)
        hfx0, qfx0 = rng.uniform(50.0, 450.0), rng.uniform(2e-5, 2.5e-4)
        pblh = rng.uniform(800.0, 2800.0)
    elif family == "stable-night":
        t = rng.uniform(268.0, 295.0)
        dtsk = -rng.uniform(1.0, 9.0)
        wind = rng.uniform(0.3, 5.0)
        hfx0, qfx0 = -rng.uniform(0.0, 60.0), rng.uniform(0.0, 1.0e-5)
        pblh = rng.uniform(30.0, 300.0)
    elif family == "neutral-windy":
        wind = rng.uniform(10.0, 25.0)
        dtsk = None   # skin theta set to the air theta below
        hfx0 = rng.uniform(-5.0, 5.0)
    elif family == "water":
        xland, mavail = 2.0, 1.0
        znt = rng.uniform(1.0e-4, 2.0e-3)
        wind = rng.choice([0.2, 1.0, 4.0, 9.0, 14.0, 20.0, 27.0, 6.0])
        dtsk = rng.uniform(-4.0, 6.0)
    elif family == "cold-land":
        t = rng.uniform(240.0, 272.0)
        dtsk = rng.uniform(-8.0, 3.0)
        rh = rng.uniform(0.5, 1.0)
        mavail = rng.uniform(0.6, 1.0)
        znt = rng.uniform(0.001, 0.05)
    elif family == "high-roughness":
        znt = rng.uniform(0.8, 2.5)
        dz = rng.uniform(60.0, 100.0)
        dtsk = rng.uniform(-6.0, 10.0)
        wind = rng.uniform(1.0, 12.0)
    elif family == "calm-convective":
        wind = [0.0, 0.0, 0.01, 0.05, 0.0, 0.2, 0.0, 0.02][n % 8]
        dtsk = rng.uniform(4.0, 15.0)
        hfx0, qfx0 = rng.uniform(100.0, 500.0), rng.uniform(5e-5, 3e-4)
        pblh = rng.uniform(1000.0, 3200.0)
    elif family == "edge":
        k = n % 8
        if k == 0:      # very stable: BR above RICUT
            t, dtsk, wind, pblh = 280.0, -15.0, 0.4, 20.0
        elif k == 1:    # |TSTV| below 1e-5 (skin theta_v at the air's)
            dtsk = None
        elif k == 2:    # weakly stable: 0 < L < 2 clamps to 2
            dtsk, wind = -6.0, 0.15
        elif k == 3:    # dry surface: MAVAIL = 0, QFX 0
            mavail = 0.0
        elif k == 4:    # skin exactly at the ice/water switch
            t, dtsk = 274.0, None
        elif k == 5:    # large downward heat flux, -250 floor
            t, dtsk, wind, znt = 300.0, -25.0, 18.0, 0.8
        elif k == 6:    # incoming negative fluxes (fluxc floors at 0)
            hfx0, qfx0 = -80.0, -5e-5
        else:           # very shallow first level
            dz = 8.0
            znt = 0.5
    elif family == "sea-ice":
        k = n % 8
        xice = [1.0, 0.95, 0.7, 0.5, 0.3, 0.15, 0.05, 0.02][k]
        t = rng.uniform(245.0, 268.0)
        sst = [271.0, 271.4, 272.5, 274.0, 276.0, 279.0, 271.2, 273.5][k]
        ice_t = t + rng.uniform(-6.0, 2.0)
        tsk_blend = xice * ice_t + (1.0 - xice) * sst
        dtsk = tsk_blend - t
        xland = 1.0 if k != 7 else 2.0
        znt, mavail = 0.001, 1.0
        wind = rng.uniform(1.0, 12.0)
        hfx0 = rng.uniform(-40.0, 40.0)
    p = psfc - 1.16 * 9.81 * dz * 0.5
    qv = rh * _qsat(t, p)
    th = t * (1.0e5 / p) ** float(RCP)
    if dtsk is None:
        # skin potential temperature equal to the air's (neutral)
        tsk = th * (psfc / 1.0e5) ** float(RCP)
        if family == "edge" and n % 8 == 4:
            tsk = 273.15
    else:
        tsk = t + dtsk
    ang = rng.uniform(0.0, 2.0 * np.pi)
    c.update(u=wind * np.cos(ang), v=wind * np.sin(ang), t=t, th=th, qv=qv,
             p=p, dz8w=dz, psfc=psfc, tsk=tsk, pblh=pblh, mavail=mavail,
             xland=xland, xice=xice)
    c["state"] = dict(regime=0.0, hfx=hfx0, qfx=qfx0, lh=0.0, mol=0.0,
                      rmol=0.0, qsfc=0.0, gz1oz0=0.0, wspd=0.0, br=0.0,
                      psim=0.0, psih=0.0, znt=znt, zol=0.0,
                      ust=[1.0e-4, 0.1, 0.3, 0.0001][n % 4], cpm=0.0,
                      chs2=0.0, cqs2=0.0, chs=0.0, flhc=0.0, flqc=0.0,
                      qgh=0.0, sst=sst)
    return c


def _columns(seed: int) -> tuple[list[dict], list[str]]:
    rng = np.random.default_rng(seed)
    cols, fams = [], []
    for family in FAMILIES:
        for n in range(PER_FAMILY):
            cols.append(_column(rng, family, n))
            fams.append(family)
    return cols, fams


def _evolve(rng, cols, fams, step: int, lsm: bool):
    """Per-step forcing and land-model overrides."""
    forcing = np.zeros((len(FORCING), len(cols)), F)
    ovr = np.zeros((len(OVERRIDE), len(cols)), F)
    for i, (c, fam) in enumerate(zip(cols, fams)):
        s = step - 1
        wscale = 1.0 + 0.08 * np.sin(0.9 * s + i)
        heat = {"unstable-day": 0.6, "stable-night": -0.4,
                "calm-convective": 0.8}.get(fam, 0.1 * np.cos(s + i))
        row = dict(c)
        row["u"] = c["u"] * wscale
        row["v"] = c["v"] * wscale
        row["tsk"] = c["tsk"] + heat * s
        if fam == "edge" and i % 8 in (1, 4):
            row["tsk"] = c["tsk"]   # keep the exact neutral / 273.15 skin
        row["pblh"] = c["pblh"] * (1.0 + 0.05 * s)
        for k, name in enumerate(FORCING):
            forcing[k, i] = F(row[name])
        if lsm and c["xland"] < 1.5 and step > 1:
            q = F(c["qv"] * (1.0 + 0.1 * rng.uniform(-1.0, 1.5)))
            if fam == "edge" and i % 8 == 6:
                q = F(-1.0e-6)    # the land model handed back QSFC <= 0
            ovr[:, i] = (1.0, q, F(c["state"]["hfx"] * (1.0 + 0.1 * s)),
                         F(abs(c["state"]["qfx"]) * (1.0 + 0.05 * s)),
                         F(c["state"]["znt"]))
    return forcing, ovr


CONFIGS = {
    "plain-dx12k": dict(mode=0, dx=12000.0, itstart=1, nstep=6, lsm=False,
                        threshold=0.5, tice=0, seed=11, warm=False),
    "plain-dx3k-lsm": dict(mode=0, dx=3000.0, itstart=1, nstep=6, lsm=True,
                           threshold=0.5, tice=0, seed=12, warm=False),
    "plain-restart": dict(mode=0, dx=1000.0, itstart=37, nstep=4, lsm=True,
                          threshold=0.5, tice=0, seed=13, warm=True),
    "ice-002": dict(mode=1, dx=12000.0, itstart=1, nstep=6, lsm=False,
                    threshold=0.02, tice=0, seed=14, warm=False),
    "ice-05-tice": dict(mode=1, dx=3000.0, itstart=1, nstep=5, lsm=True,
                        threshold=0.5, tice=1, seed=15, warm=False),
}


def write_config(out: Path, name: str, spec: dict) -> dict:
    cols, fams = _columns(spec["seed"])
    rng = np.random.default_rng(spec["seed"] + 1000)
    ncol = len(cols)
    state = np.zeros((len(STATE), ncol), F)
    for i, c in enumerate(cols):
        st = dict(c["state"])
        if spec["warm"]:
            # A warm restart: the land model's QSFC, last step's u* and
            # fluxes, a nonzero MOL/RMOL, so ITIMESTEP > 1 reads them.
            st.update(qsfc=c["qv"] * 1.05, ust=0.25, mol=0.1,
                      rmol=0.002, regime=2.0)
            if i % 9 == 4:
                st["qsfc"] = 0.0          # QSFC <= 0: recomputed
        for k, field in enumerate(STATE):
            state[k, i] = F(st[field])
    path = out / f"{name}.in.bin"
    with path.open("wb") as fh:
        fh.write(struct.pack("<8i", MAGIC_IN, 1, ncol, spec["nstep"],
                             spec["mode"], spec["itstart"], spec["tice"], 0))
        fh.write(struct.pack("<2f", spec["dx"], spec["threshold"]))
        # Fortran reads state(ncol, NSTATE): column-major, field-major blocks
        fh.write(state.astype("<f4").tobytes(order="C"))
        for step in range(1, spec["nstep"] + 1):
            forcing, ovr = _evolve(rng, cols, fams, step, spec["lsm"])
            fh.write(forcing.astype("<f4").tobytes(order="C"))
            fh.write(ovr.astype("<f4").tobytes(order="C"))
    return {"file": path.name, "families": fams, **{k: v for k, v in
            spec.items() if k not in ("seed",)}, "seed": spec["seed"]}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: make_cases.py OUT_DIR", file=sys.stderr)
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    index = {name: write_config(out, name, spec)
             for name, spec in CONFIGS.items()}
    (out / "cases-index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {len(index)} configurations to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
