"""Captured WRF MYJSFC inputs -> run_myjsfc CASE.bin, and OUT.bin -> word counts.

  python sfc_case.py write CAPDIR STEP CASE.bin
  python sfc_case.py compare CAPDIR STEP OUT.bin [GPUWM.npz]
"""
import sys

import numpy as np

from myj_replay import assemble, words

INOUT = ("ust", "znt", "thz0", "qz0", "uz0", "vz0", "qsfc", "akhs", "akms")
OUTS = ("rmol", "ct", "pblh", "br", "chs", "chs2", "cqs2", "hfx", "qfx", "lh",
        "flhc", "flqc", "qgh", "cpm", "u10", "v10", "t2", "th2", "tshltr", "th10",
        "q2", "qshltr", "q10", "pshltr", "u10e", "v10e")
NZ = 44


def write(capdir, step, path):
    win, _ = assemble(capdir, "myjsfc", "in", step)
    ncol = win["tsk"].size
    with open(path, "wb") as fh:
        np.array([ncol, NZ, 1, int(win["itimestep"])], "<i4").tofile(fh)
        win["ht"].astype("<f4").ravel().tofile(fh)
        for n in ("dz8w", "p_phy", "th_phy", "t_phy", "qv_curr", "qc_curr", "u_phy", "v_phy", "tke_pbl"):
            win[n][:NZ].reshape(NZ, ncol).astype("<f4").tofile(fh)
        win["p8w"][:NZ + 1].reshape(NZ + 1, ncol).astype("<f4").tofile(fh)
        for n in ("tsk", "xland", "mavail", "z0", "qsfc", "thz0", "qz0", "uz0", "vz0", "ust", "znt", "akhs", "akms"):
            win[n].astype("<f4").ravel().tofile(fh)
    print("wrote", path, ncol)


def read_out(path):
    raw = open(path, "rb").read()
    ncol, nz, nsteps, it0, kztm, nfield = np.frombuffer(raw, "<i4", 6, 0)
    pos = 24 + 8 * 4 + 4 * kztm * 4
    data = np.frombuffer(raw, "<f4", nfield * ncol, pos).reshape(nfield, ncol)
    return dict(zip(INOUT + OUTS, data))


def compare(capdir, step, path, other=None):
    wout, _ = assemble(capdir, "myjsfc", "out", step)
    got = read_out(path)
    tot = 0
    for n, a in got.items():
        d, u = words(a, wout[n].ravel())
        tot += d
        if d:
            print(f"   {n:8s} {d:8d}  max {u} ulp")
    print("vs WRF capture: words different", tot)
    if other:
        g = np.load(other)
        tot = 0
        for n, a in got.items():
            k = "rib" if n == "br" else n
            d, u = words(a, g[k].ravel())
            tot += d
        print("vs woof:       words different", tot)


if __name__ == "__main__":
    if sys.argv[1] == "write":
        write(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        compare(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
