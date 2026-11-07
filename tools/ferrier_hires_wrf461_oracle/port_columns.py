#!/usr/bin/env python3
"""Run WOOF's Ferrier-Aligo kernel over an oracle column set.

usage: port_columns.py COLUMNS_INPUTS_CSV OUT_DIR

Reads a ``columns_inputs.csv`` written by ``make_columns.py`` /
``refute/make_columns_refute.py`` (mp_physics=5, fields ending
``f_ice,f_rain,f_rimef``) or by ``advect/make_columns_advect.py``
(mp_physics=15, a ``qrimef`` field), drives ``woof.core.fer_hires``
exactly as the matching Fortran driver drives WRF (init at the column's
time step, then NSTEPS calls carrying the package's own state), and
writes ``levels.csv`` / ``surface.csv`` in the Fortran driver's format so
``compare_ferrier.py`` grades them bit for bit.  Needs a CUDA device.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _f32(h: str) -> np.float32:
    return np.uint32(int(h, 16)).view(np.float32)


def _hex(x) -> str:
    return f"{int(np.float32(x).view(np.uint32)):08X}"


def read_columns(path: Path):
    fields = None
    cols = []
    for line in path.read_text().splitlines():
        if line.startswith("# level rows"):
            fields = line.split("L,col,k,", 1)[1].split(",")
        elif line.startswith("C,"):
            _, idx, regime, gid, nsteps, dt, dx, rain = line.split(",")
            cols.append({"idx": int(idx), "regime": regime, "gid": int(gid),
                         "nsteps": int(nsteps), "dt": _f32(dt),
                         "rainnc": _f32(rain), "levels": []})
        elif line.startswith("L,"):
            parts = line.split(",")
            cols[-1]["levels"].append([_f32(h) for h in parts[3:]])
    return fields, cols


def run(inputs: Path, out: Path, kernel=None) -> None:
    import cupy as cp
    from woof.core import fer_hires as fh

    fields, cols = read_columns(inputs)
    variant = "advected_rime" if "qrimef" in fields else "state_fractions"
    out.mkdir(parents=True, exist_ok=True)
    if variant == "advected_rime":
        lev_head = "col,step,k,th,qv,qc,qr,qi,qrimef"
    else:
        lev_head = "col,step,k,th,qv,qt,qc,qr,qi,f_ice,f_rain,f_rimef"
    with open(out / "levels.csv", "w", newline="\n") as fl, \
            open(out / "surface.csv", "w", newline="\n") as fs:
        fl.write(lev_head + "\n")
        fs.write("col,step,rainnc,rainncv,sr\n")
        for ic, col in enumerate(cols):
            a = np.array(col["levels"], dtype=np.float32)   # (nz, nfields)
            nz = a.shape[0]
            g = {name: cp.asarray(np.ascontiguousarray(
                a[:, j].reshape(nz, 1, 1))) for j, name in enumerate(fields)}
            rainnc = cp.full((1, 1), col["rainnc"], dtype=cp.float32)
            rainncv = cp.zeros((1, 1), dtype=cp.float32)
            sr = cp.zeros((1, 1), dtype=cp.float32)
            no_fmad = bool(fh.package_for(15 if variant == "advected_rime" else 5).get("no_fmad", False))
            tab = fh.device_tables(float(col["dt"]), no_fmad=no_fmad)
            extra = {}
            if variant == "advected_rime":
                extra["qrimef"] = g["qrimef"]
            else:
                extra["qt"] = cp.zeros_like(g["th"])
                extra["f_ice"] = g["f_ice"]
                extra["f_rain"] = g["f_rain"]
                extra["f_rimef"] = g["f_rimef"]
            for step in range(1, col["nsteps"] + 1):
                unused = g["qi"]
                fh.launch_fer_hires(g["th"], g["qv"], g["qc"], g["qr"], g["qi"],
                                   extra.get("qt", unused),
                                   extra.get("f_ice", unused),
                                   extra.get("f_rain", unused),
                                   extra.get("f_rimef", unused),
                                   g["p"], g["pi"], g["rho"], g["dz8w"],
                                   rainnc, rainncv, sr, tab,
                                   dt=float(col["dt"]), gid=col["gid"],
                                   qrimef=extra.get("qrimef"), kernel=kernel,
                                   no_fmad=no_fmad)
                if step == 1 or step == col["nsteps"]:
                    if variant == "advected_rime":
                        names = ("th", "qv", "qc", "qr", "qi", "qrimef")
                        src = {**g}
                    else:
                        names = ("th", "qv", "qt", "qc", "qr", "qi",
                                 "f_ice", "f_rain", "f_rimef")
                        src = {**g, **extra}
                    host = {n: src[n].get().reshape(nz) for n in names}
                    for k in range(nz):
                        fl.write(f"{ic},{step},{k + 1},"
                                 + ",".join(_hex(host[n][k]) for n in names)
                                 + "\n")
                    fs.write(f"{ic},{step},{_hex(rainnc.get()[0, 0])},"
                             f"{_hex(rainncv.get()[0, 0])},"
                             f"{_hex(sr.get()[0, 0])}\n")
    print(f"wrote {len(cols)} columns ({variant}) to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run(Path(sys.argv[1]), Path(sys.argv[2]))
