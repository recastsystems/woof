"""Replay WRF 4.6.1's captured MYJSFC / MYJPBL wrapper inputs through woof's kernels.

The capture build (wrfcap/patch_drivers.py) writes, for every tile and every
requested step, each wrapper argument immediately before the CALL ("in") and
the INOUT/OUT arguments immediately after it ("out").  This script assembles
the tiles into full-domain arrays, feeds the "in" set to woof's launchers
exactly as woof/core/physics.py binds them, and counts differing float32
words against WRF's "out" set.

usage: python myj_replay.py CAPDIR STEP [STEP ...] --json OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

HDR = np.dtype([("name", "S16"), ("kind", "<i4"), ("i0", "<i4"), ("i1", "<i4"),
                ("k0", "<i4"), ("k1", "<i4"), ("j0", "<i4"), ("j1", "<i4")])


def read_tile(path):
    raw = open(path, "rb").read()
    pos, out = 0, {}
    while pos < len(raw):
        h = np.frombuffer(raw, HDR, 1, pos)[0]
        pos += HDR.itemsize
        name = h["name"].decode().strip()
        kind = int(h["kind"])
        ni, nk, nj = h["i1"] - h["i0"] + 1, h["k1"] - h["k0"] + 1, h["j1"] - h["j0"] + 1
        n = int(ni * nk * nj)
        dt = "<i4" if kind in (4, 5) else "<f4"
        a = np.frombuffer(raw, dt, n, pos).copy()
        pos += 4 * n
        if kind == 3:
            a = a.reshape(nj, nk, ni)
        elif kind in (2, 5):
            a = a.reshape(nj, ni)
        else:
            a = a[0]
        out[name] = (kind, (int(h["i0"]), int(h["j0"]), int(h["k0"])), a)
    return out


def assemble(capdir, tag, phase, step):
    files = sorted(glob.glob(os.path.join(capdir, f"{tag}_{phase}_s{step:05d}_i*_j*.bin")))
    if not files:
        raise SystemExit(f"no capture for {tag} {phase} step {step} in {capdir}")
    tiles = [read_tile(f) for f in files]
    ni = max(t[n][1][0] + t[n][2].shape[-1] - 1 for t in tiles for n in t if t[n][0] in (2, 3, 5))
    nj = max(t[n][1][1] + t[n][2].shape[0] - 1 for t in tiles for n in t if t[n][0] in (2, 3, 5))
    full = {}
    for t in tiles:
        for name, (kind, (i0, j0, k0), a) in t.items():
            if kind in (1, 4):
                full.setdefault(name, a)
                continue
            if name not in full:
                if kind == 3:
                    full[name] = np.full((a.shape[1], nj, ni), np.nan, np.float32)
                else:
                    full[name] = np.zeros((nj, ni), np.int32 if kind == 5 else np.float32)
            if kind == 3:
                full[name][:, j0 - 1:j0 - 1 + a.shape[0], i0 - 1:i0 - 1 + a.shape[2]] = a.transpose(1, 0, 2)
            else:
                full[name][j0 - 1:j0 - 1 + a.shape[0], i0 - 1:i0 - 1 + a.shape[1]] = a
    return full, len(files)


def words(a, b):
    """Differing float32 (or int32) words and the largest ULP distance."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.dtype == np.int32:
        d = a != b
        return int(d.sum()), int(np.abs(a.astype(np.int64) - b).max()) if d.any() else 0
    ua, ub = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    # total-order map so the ULP distance is a subtraction
    oa = np.where(ua < 0, -(ua & 0x7FFFFFFF), ua)
    ob = np.where(ub < 0, -(ub & 0x7FFFFFFF), ub)
    d = ua != ub
    return int(d.sum()), int(np.abs(oa - ob)[d].max()) if d.any() else 0


def replay_sfc(cp, capdir, step, nz):
    from woof.core.myjsfc import launch_myj_sfclay
    from woof.core.physics_inventory import MYJ_SFCLAY_INOUT, MYJ_SFCLAY_OUTPUTS
    win, ntile = assemble(capdir, "myjsfc", "in", step)
    wout, _ = assemble(capdir, "myjsfc", "out", step)
    D = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))
    columns = {"dz": D(win["dz8w"][:nz]), "tke": D(win["tke_pbl"][:nz])}
    surface = {"u1": D(win["u_phy"][0]), "v1": D(win["v_phy"][0]),
               "t1": D(win["t_phy"][0]), "th1": D(win["th_phy"][0]),
               "qv1": D(win["qv_curr"][0]), "qc1": D(win["qc_curr"][0]),
               "p1": D(win["p_phy"][0]), "psfc": D(win["p8w"][0]),
               "tsk": D(win["tsk"]), "xland": D(win["xland"]),
               "mavail": D(win["mavail"]), "z0base": D(win["z0"]),
               "ht": D(win["ht"])}
    state = {n: D(win[n]) for n in MYJ_SFCLAY_INOUT}
    seed = {"rib": "br"}
    outputs = {n: D(win.get(seed.get(n, n), np.zeros_like(win["tsk"]))) for n in MYJ_SFCLAY_OUTPUTS}
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    launch_myj_sfclay(columns, surface, state, outputs, itimestep=int(win["itimestep"]))
    cp.cuda.Device().synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    rows = {}
    for n in list(MYJ_SFCLAY_INOUT) + list(MYJ_SFCLAY_OUTPUTS):
        ref = wout[seed.get(n, n)]
        got = cp.asnumpy(state[n] if n in state else outputs[n])
        rows[n] = dict(zip(("diff", "max_ulp"), words(got, ref)), n=int(ref.size))
    return rows, ntile, ms, win, wout


def replay_pbl(cp, capdir, step, nz):
    from woof.core.myjpbl import MYJ_PBL_INOUT, myj_pbl_step
    win, ntile = assemble(capdir, "myjpbl", "in", step)
    wout, _ = assemble(capdir, "myjpbl", "out", step)
    D = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))
    flqi = "qi_curr" in win
    columns = {"dz": D(win["dz8w"][:nz]), "u": D(win["u_phy"][:nz]),
               "v": D(win["v_phy"][:nz]), "t": D(win["t_phy"][:nz]),
               "th": D(win["th_phy"][:nz]), "exner": D(win["pi_phy"][:nz]),
               "qv": D(win["qv_curr"][:nz]), "qc": D(win["qc_curr"][:nz]),
               "qi": D(win["qi_curr"][:nz]) if flqi else None,
               "p": D(win["p_phy"][:nz])}
    surface = {"psfc": D(win["p8w"][0]), "ust": D(win["ust"]), "tsk": D(win["tsk"]),
               "chklowq": D(win["chklowq"]), "xland": D(win["xland"]),
               "sice": D(win["xice"]), "snow": D(win["snow"]),
               "akhs": D(win["akhs"]), "akms": D(win["akms"]), "elflx": D(win["lh"]),
               "uz0": D(win["uz0"]), "vz0": D(win["vz0"]), "ht": D(win["ht"])}
    state = {n: D(win[n]) for n in MYJ_PBL_INOUT}
    tke = D(win["tke_pbl"][:nz])
    dtturbl = float(np.float32(win["dt"]) * np.float32(win["stepbl"]))
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = myj_pbl_step(columns, surface, state, tke, dtturbl=dtturbl, flqi=flqi)
    cp.cuda.Device().synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    pairs = [("tke_myj", tke, "tke_pbl"), ("exch_h", out["exch_h"], "exch_h"),
             ("el_myj", out["el_myj"], "el_pbl")]
    pairs += [(n, out[n], n) for n in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten")]
    if flqi:
        pairs.append(("rqiblten", out["rqiblten"], "rqiblten"))
    rows = {}
    for n, got, ref in pairs:
        rows[n] = dict(zip(("diff", "max_ulp"), words(cp.asnumpy(got), wout[ref][:nz])),
                       n=int(wout[ref][:nz].size))
    for n in ("pblh", "mixht", "kpbl"):
        rows[n] = dict(zip(("diff", "max_ulp"), words(cp.asnumpy(out[n]), wout[n])), n=int(wout[n].size))
    for n in MYJ_PBL_INOUT:
        rows[n] = dict(zip(("diff", "max_ulp"), words(cp.asnumpy(state[n]), wout[n])), n=int(wout[n].size))
    # INOUT words WRF's MYJPBL may rewrite that woof's launcher treats as read-only
    for n in ("ust", "znt", "uz0", "vz0", "akhs", "akms"):
        rows[f"{n} (wrf in->out)"] = dict(zip(("diff", "max_ulp"), words(win[n], wout[n])), n=int(wout[n].size))
    return rows, ntile, ms, flqi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capdir")
    ap.add_argument("steps", nargs="+", type=int)
    ap.add_argument("--json", required=True)
    ap.add_argument("--nz", type=int, default=44)
    args = ap.parse_args()
    import cupy as cp
    from woof import numerics, wrf_exact
    receipt = {"capdir": os.path.abspath(args.capdir), "numerics": numerics.mode(),
               "wrf_exact": wrf_exact.ENABLED, "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               "steps": {}}
    total = 0
    for step in args.steps:
        srows, nt, sms, win, wout = replay_sfc(cp, args.capdir, step, args.nz)
        prows, _, pms, flqi = replay_pbl(cp, args.capdir, step, args.nz)
        sd = sum(r["diff"] for r in srows.values())
        pd = sum(r["diff"] for k, r in prows.items() if "wrf in->out" not in k)
        total += sd + pd
        receipt["steps"][str(step)] = {"tiles": nt, "myjsfc": srows, "myjsfc_ms": sms,
                                       "myjsfc_words_different": sd, "myjpbl": prows,
                                       "myjpbl_ms": pms, "myjpbl_words_different": pd, "flqi": flqi}
        print(f"step {step}: tiles {nt}  MYJSFC {sd} words different  MYJPBL {pd} words different")
        for k, r in list(srows.items()) + list(prows.items()):
            if r["diff"]:
                print(f"    {k:24s} {r['diff']:>10d} / {r['n']:<10d} max {r['max_ulp']} ulp")
    receipt["words_different_total"] = total
    json.dump(receipt, open(args.json, "w"), indent=1)
    print("TOTAL words different:", total)


if __name__ == "__main__":
    main()
