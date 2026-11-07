#!/usr/bin/env python3
"""Write the column cases the GFS surface-layer oracle runs (cases.bin).

The case set is the input half of the fixture of record, so it is generated
here, deterministically and with no random numbers, and checked in beside
the oracle's outputs.  Every column is a named regime; the per-step inputs
drift smoothly so the carried ZNT (Charnock roughness over water), the
carried UST and the wrapper's SST are exercised across steps rather than
re-seeded each call.

Layout (little-endian):

    int32  magic 0x47465331 ("GFS1"), ncol, nstep, nin, ninit
    float32 per_step[nstep][nin][ncol]    (STEP_FIELDS order)
    float32 initial[ninit][ncol]          (INITIAL_FIELDS order)

Usage::

    python make_cases.py OUT_FILE
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = 0x47465331
#: Per-step inputs SF_GFS reads (level-1 atmosphere, surface, masks).
STEP_FIELDS = ("u1", "v1", "t1", "qv1", "p1", "psfc", "tsk", "xland", "xice")
#: State seeded once and carried by the scheme or the wrapper afterwards.
INITIAL_FIELDS = ("znt", "ust", "sst")
NSTEP = 6

F = np.float32


def _base_cases() -> list[dict]:
    """One dict per column: the regime's first-step values."""
    cases: list[dict] = []

    def add(name, *, u, v, t1, qv, p1, psfc, tsk, xland, znt,
            xice=0.0, sst=None, ust=0.1):
        cases.append(dict(name=name, u1=u, v1=v, t1=t1, qv1=qv, p1=p1,
                          psfc=psfc, tsk=tsk, xland=xland, xice=xice,
                          znt=znt, ust=ust,
                          sst=tsk if sst is None else sst))

    # Unstable land: surface warmer than the air, a spread of winds and
    # roughness lengths, low and deep first layers.
    for k, (wind, z0, dt) in enumerate((
            (0.6, 0.05, 6.0), (2.0, 0.10, 8.0), (4.5, 0.25, 4.0),
            (7.0, 0.50, 3.0), (11.0, 0.80, 2.0), (15.0, 0.03, 1.5),
            (3.0, 0.01, 12.0), (1.2, 0.15, 15.0))):
        add(f"land-unstable-{k}", u=wind * 0.8, v=-wind * 0.6,
            t1=298.0 - k, qv=0.012 - 0.001 * k, p1=99050.0 - 40.0 * k,
            psfc=100000.0 - 30.0 * k, tsk=298.0 - k + dt, xland=1.0,
            znt=z0)
    # Stable land: night inversions over a broad Ri range, so both the
    # HLINF > 0.25 iteration and the weakly stable pass are reached.
    for k, (wind, z0, dt) in enumerate((
            (0.4, 0.10, -6.0), (1.5, 0.20, -4.0), (3.0, 0.05, -2.0),
            (6.0, 0.40, -1.0), (9.0, 0.10, -0.3), (2.5, 0.70, -8.0),
            (5.0, 0.02, -0.05), (12.0, 0.30, -3.0))):
        add(f"land-stable-{k}", u=-wind * 0.5, v=wind * 0.866,
            t1=283.0 + k, qv=0.006 + 0.0005 * k, p1=99200.0 - 50.0 * k,
            psfc=100100.0 - 40.0 * k, tsk=283.0 + k + dt, xland=1.0,
            znt=z0)
    # Open sea: Charnock roughness carried across steps, both stabilities,
    # trade-wind to gale speeds.
    for k, (wind, z0, dt) in enumerate((
            (1.0, 1.0e-4, 1.0), (5.0, 2.0e-4, 0.5), (9.0, 5.0e-4, 1.5),
            (14.0, 1.0e-3, 2.0), (22.0, 2.0e-3, 0.8), (28.0, 3.0e-3, 1.2),
            (4.0, 1.0e-4, -1.5), (8.0, 3.0e-4, -0.7), (16.0, 1.0e-3, -2.0),
            (0.3, 1.0e-4, 3.0))):
        add(f"sea-{k}", u=wind * 0.9, v=wind * 0.436,
            t1=300.0 - 0.8 * k, qv=0.018 - 0.0009 * k,
            p1=100100.0 - 15.0 * k, psfc=101100.0 - 10.0 * k,
            tsk=300.0 - 0.8 * k + dt, xland=2.0, znt=z0, ust=0.2)
    # Calm: zero and near-zero winds (the 1 m/s floor) over land and sea.
    add("calm-land-zero", u=0.0, v=0.0, t1=290.0, qv=0.009, p1=99000.0,
        psfc=99950.0, tsk=293.0, xland=1.0, znt=0.1)
    add("calm-land-tiny", u=0.05, v=-0.02, t1=290.0, qv=0.009, p1=99000.0,
        psfc=99950.0, tsk=287.0, xland=1.0, znt=0.1)
    add("calm-sea-zero", u=0.0, v=0.0, t1=295.0, qv=0.015, p1=100300.0,
        psfc=101300.0, tsk=296.5, xland=2.0, znt=1.0e-4)
    add("calm-sea-stable", u=0.2, v=0.1, t1=295.0, qv=0.015, p1=100300.0,
        psfc=101300.0, tsk=293.0, xland=2.0, znt=1.0e-4)
    # Rough: forest and urban roughness against a shallow first layer, so
    # Z0MAX = MIN(Z0, 0.1*Z1) binds.
    add("rough-forest", u=6.0, v=2.0, t1=295.0, qv=0.011, p1=99800.0,
        psfc=100000.0, tsk=298.0, xland=1.0, znt=2.0)
    add("rough-urban", u=3.0, v=-4.0, t1=300.0, qv=0.008, p1=99880.0,
        psfc=100000.0, tsk=306.0, xland=1.0, znt=2.65)
    add("rough-stable", u=2.0, v=1.0, t1=285.0, qv=0.005, p1=99800.0,
        psfc=100000.0, tsk=280.0, xland=1.0, znt=1.5)
    # Very unstable: a hot dry surface under light wind drives HLINF below
    # -0.5 and into the unphysical-Obukhov clamp.
    add("desert-hot", u=0.8, v=0.4, t1=308.0, qv=0.002, p1=89000.0,
        psfc=90000.0, tsk=331.0, xland=1.0, znt=0.01)
    add("desert-beyond-table", u=1.5, v=0.0, t1=312.0, qv=0.001,
        p1=88900.0, psfc=90000.0, tsk=333.5, xland=1.0, znt=0.005)
    # Elevated terrain, cold and dry: the fpvs ice and mixed branches.
    add("plateau-cold", u=5.0, v=5.0, t1=250.0, qv=4.0e-4, p1=69300.0,
        psfc=70000.0, tsk=252.0, xland=1.0, znt=0.05)
    add("arctic-land", u=3.0, v=-1.0, t1=232.0, qv=5.0e-5, p1=100500.0,
        psfc=101500.0, tsk=228.0, xland=1.0, znt=0.01)
    add("mixed-phase", u=4.0, v=1.0, t1=262.0, qv=1.5e-3, p1=94000.0,
        psfc=95000.0, tsk=266.0, xland=1.0, znt=0.05)
    add("bone-dry", u=7.0, v=-3.0, t1=296.0, qv=1.0e-9, p1=99000.0,
        psfc=100000.0, tsk=300.0, xland=1.0, znt=0.1)
    add("tropical-moist", u=2.0, v=1.0, t1=302.0, qv=0.024, p1=100200.0,
        psfc=101200.0, tsk=304.0, xland=1.0, znt=0.5)
    # Sea ice.  WRF's real.exe marks a cell at or above the ice threshold
    # as land (XLAND=1); the XLAND=2 rows exercise the mask the open-water
    # arm of the fractional wrapper is meant to impose either way.  SST
    # above 273 K at ITIMESTEP <= 3 reaches get_local_ice_tsk's
    # first-steps adjustment ladder (module_surface_driver.F:7167-7178).
    for k, (xice, xland, tsk, sst, wind, t1) in enumerate((
            (1.00, 1.0, 250.0, 271.4, 6.0, 252.0),
            (0.95, 1.0, 252.0, 271.4, 9.0, 255.0),
            (0.70, 1.0, 258.0, 274.5, 4.0, 262.0),
            (0.50, 1.0, 262.0, 276.0, 7.0, 260.0),
            (0.30, 1.0, 266.0, 279.0, 3.0, 268.0),
            (0.15, 1.0, 250.0, 271.0, 11.0, 255.0),
            (0.05, 1.0, 258.0, 271.4, 5.0, 262.0),
            (0.03, 1.0, 270.0, 280.0, 2.0, 266.0),
            (0.60, 2.0, 262.0, 271.6, 8.0, 258.0),
            (0.25, 2.0, 268.0, 272.0, 12.0, 264.0),
            (0.01, 2.0, 271.0, 271.5, 6.0, 268.0),
            (0.40, 1.0, 245.0, 271.4, 15.0, 248.0))):
        add(f"seaice-{k}", u=wind * 0.7, v=-wind * 0.714, t1=t1,
            qv=6.0e-4 + 1.0e-4 * k, p1=100400.0 - 20.0 * k,
            psfc=101400.0 - 15.0 * k, tsk=tsk, xland=xland,
            znt=(1.0e-4 if xland == 2.0 else 1.0e-3), xice=xice, sst=sst)
    # Mixed regimes between the named ones: gales, near-neutral, a
    # 20 K superadiabatic skin, a 12 K inversion.
    for k, (xl, wind, dt, z0) in enumerate((
            (1.0, 8.0, 0.0, 0.2), (2.0, 10.0, 0.0, 6.0e-4),
            (1.0, 0.9, 20.0, 0.08), (2.0, 18.0, -4.0, 1.5e-3),
            (1.0, 25.0, 0.5, 0.6), (2.0, 2.0, 6.0, 1.0e-4),
            (1.0, 4.0, -12.0, 0.12), (2.0, 7.0, 0.01, 2.0e-4),
            (1.0, 13.0, -0.6, 0.35), (2.0, 32.0, 1.0, 4.0e-3),
            (1.0, 0.7, -0.2, 0.9), (2.0, 0.6, -0.4, 1.0e-4),
            (1.0, 5.5, 2.2, 0.04), (2.0, 12.0, 2.5, 8.0e-4))):
        add(f"mixed-{k}", u=wind * math.cos(0.4 * k),
            v=wind * math.sin(0.4 * k), t1=288.0 + k, qv=0.008,
            p1=99400.0 - 120.0 * k, psfc=100000.0 - 100.0 * k,
            tsk=288.0 + k + dt, xland=xl, znt=z0)
    return cases


def build() -> tuple[np.ndarray, np.ndarray]:
    cases = _base_cases()
    ncol = len(cases)
    per_step = np.zeros((NSTEP, len(STEP_FIELDS), ncol), dtype=F)
    for step in range(NSTEP):
        for i, case in enumerate(cases):
            # Smooth, column-dependent drift: winds by up to 25 %, air and
            # skin temperatures by fractions of a kelvin, so stability and
            # u* move between steps without changing regime.
            phase = 0.9 * step + 0.37 * i
            wind_scale = 1.0 + 0.25 * math.sin(phase)
            dtemp = 0.3 * math.sin(0.7 * step + 0.11 * i)
            dskin = 0.4 * math.cos(0.5 * step + 0.23 * i)
            row = {
                "u1": case["u1"] * wind_scale,
                "v1": case["v1"] * wind_scale,
                "t1": case["t1"] + dtemp,
                "qv1": case["qv1"] * (1.0 + 0.02 * math.sin(phase)),
                "p1": case["p1"] - 3.0 * step,
                "psfc": case["psfc"] - 3.0 * step,
                "tsk": case["tsk"] + dskin,
                "xland": case["xland"],
                "xice": case["xice"],
            }
            for f, name in enumerate(STEP_FIELDS):
                per_step[step, f, i] = F(row[name])
    initial = np.zeros((len(INITIAL_FIELDS), ncol), dtype=F)
    for i, case in enumerate(cases):
        for f, name in enumerate(INITIAL_FIELDS):
            initial[f, i] = F(case[name])
    return per_step, initial


def case_names() -> list[str]:
    return [case["name"] for case in _base_cases()]


def write(path: Path) -> None:
    per_step, initial = build()
    nstep, nin, ncol = per_step.shape
    with open(path, "wb") as handle:
        handle.write(struct.pack("<5i", MAGIC, ncol, nstep, nin,
                                 initial.shape[0]))
        handle.write(np.ascontiguousarray(per_step, dtype="<f4").tobytes())
        handle.write(np.ascontiguousarray(initial, dtype="<f4").tobytes())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    write(Path(sys.argv[1]))
    print(f"{len(case_names())} columns x {NSTEP} steps -> {sys.argv[1]}")
