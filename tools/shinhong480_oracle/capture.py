#!/usr/bin/env python3
"""Turn native WRF 4.8.0 Shin-Hong captures into v2 column records.

    python -m tools.shinhong480_oracle.capture convert CAPDIR IN.bin INRUN-OUT.bin

CAPDIR holds the shinhong-<step>-<its>-<jts>-{in,out}.bin files the
instrumented copy writes (instrument_pbl_driver.py).  Every column of every
captured tile and step becomes one v2 record (columns.py layout); the
native in-run outputs are written beside them in the driver's output layout,
so the column driver's own replay of the same inputs can be checked against
what the running model computed.  t_phy is checked to be th_phy*pi_phy in
float32 on every word (the engine forms it that way).
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

from tools.shinhong480_oracle.columns import (WRF_LEVELS, WRF_SURFACE,
                                              write_v2)

F32 = np.float32
IN3 = ("u", "v", "theta", "t", "qv", "qc", "qi", "p", "pi", "dz", "tke",
       "rthraten")
IN2 = ("psfc", "znt", "ust", "psim", "psih", "xland", "hfx", "qfx", "wspd",
       "br", "u10", "v10", "uoce", "voce", "ctopo", "ctopo2", "dx", "corf")
OUT3 = ("du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "exch_m", "tke",
        "el")
OUT2 = ("hpbl", "wstar", "delta", "u10", "v10")


def _read_in(path):
    b = path.read_bytes()
    h = struct.unpack_from("<10i", b, 0)
    nx, ny, nz, step, fqc, fqi, nlf, scu, dissip, _ = h
    o = 40
    dt = struct.unpack_from("<f", b, o)[0]
    o += 4
    d = {}
    for name in IN3:
        n = nx * nz * ny
        d[name] = np.frombuffer(b, "<f4", n, o).reshape((ny, nz, nx)); o += 4 * n
    n = nx * (nz + 1) * ny
    d["p_interface"] = np.frombuffer(b, "<f4", n, o).reshape((ny, nz + 1, nx)); o += 4 * n
    for name in IN2:
        d[name] = np.frombuffer(b, "<f4", nx * ny, o).reshape((ny, nx)); o += 4 * nx * ny
    assert o == len(b), (path, o, len(b))
    flags = (1 if nlf else 0) | (2 if scu else 0) | (4 if dissip else 0)
    return dict(nx=nx, ny=ny, nz=nz, step=step, fqc=fqc, fqi=fqi, dt=F32(dt),
                flags=flags, fields=d)


def _read_out(path, nx, ny, nz):
    b = path.read_bytes()
    assert struct.unpack_from("<4i", b, 0)[:3] == (nx, ny, nz)
    o = 16
    d = {}
    for name in OUT3:
        n = nx * nz * ny
        d[name] = np.frombuffer(b, "<f4", n, o).reshape((ny, nz, nx)); o += 4 * n
    for name in OUT2:
        d[name] = np.frombuffer(b, "<f4", nx * ny, o).reshape((ny, nx)); o += 4 * nx * ny
    d["kpbl"] = np.frombuffer(b, "<i4", nx * ny, o).reshape((ny, nx)); o += 4 * nx * ny
    assert o == len(b)
    return d


def convert(capdir: Path, out_in: Path, out_run: Path):
    files = sorted(capdir.glob("shinhong-*-in.bin"),
                   key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.name)))
    recs, inrun = [], []
    mism = 0
    for f in files:
        c = _read_in(f)
        o = _read_out(f.with_name(f.name.replace("-in.bin", "-out.bin")),
                      c["nx"], c["ny"], c["nz"])
        d = c["fields"]
        mism += int(((d["theta"] * d["pi"]).astype(F32).view(np.uint32)
                     != d["t"].view(np.uint32)).sum())
        for j in range(c["ny"]):
            for i in range(c["nx"]):
                r = {"nz": c["nz"], "flags": c["flags"], "family": c["step"],
                     "base": 0, "topo": 0}
                for name in IN3:
                    r[name] = np.array(d[name][j, :, i], F32)
                if not c["fqc"]:
                    r["qc"] = np.zeros(c["nz"], F32)
                if not c["fqi"]:
                    r["qi"] = np.zeros(c["nz"], F32)
                r["p_interface"] = np.array(d["p_interface"][j, :, i], F32)
                s = {name: F32(d[name][j, i]) for name in IN2}
                s["dt"] = c["dt"]
                s["dy"] = s["dx"]
                r["scalars"] = s
                if s["ctopo"] != 1 or s["ctopo2"] != 1:
                    r["topo"] = 1
                recs.append(r)
                inrun.append({"kpbl": int(o["kpbl"][j, i]),
                              **{n: o[n][j, :, i] for n in OUT3},
                              **{n: o[n][j, i] for n in OUT2}})
    write_v2(recs, out_in)
    with open(out_run, "wb") as fh:
        for r, d in zip(recs, inrun):
            fh.write(struct.pack("<4i", r["nz"], d["kpbl"], r["family"], 0))
            for name in WRF_LEVELS:
                fh.write(np.asarray(d[name], "<f4").tobytes())
            fh.write(np.asarray([d[n] for n in WRF_SURFACE], "<f4").tobytes())
    print(f"{len(files)} tiles, {len(recs)} columns; t_phy != th_phy*pi_phy "
          f"on {mism} words")
    return recs


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "convert":
        print(__doc__)
        sys.exit(2)
    convert(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
