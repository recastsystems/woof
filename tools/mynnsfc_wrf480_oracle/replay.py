"""Module replay: WRF v4.8.0 MYNN-SFC (b66187f5) on the GPU against the
native strict referee, word for word.

A capture is an ``.npz`` holding the actual arguments of one WRF surface-layer
call site just before the call (``N3`` two-level profiles, ``N2`` surface
fields, the header options).  ``run`` hands the same words to

* ``replay_mynnsfc.exe`` (built by ``build.sh`` against the referee's own
  ``libwrflib.a``), which calls ``mynnsfc_driver`` or ``mynn_seaice_wrapper``;
* WOOF-ArWen's ``Physics._run_mynnsfc_b66187f`` (the forecast path itself, on a
  stand-in owner holding exactly these fields),

and compares every output word.  Usage::

    python replay.py compare CAPTURE.npz --native ./replay_mynnsfc.exe
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

N3 = ("u", "v", "temperature", "qv", "pressure", "dz", "theta", "rho",
      "pattern_spp_pbl")
N2 = ("psfc", "mavail", "xland", "tsk", "snowh", "xice", "dx", "water_depth",
      "lakemask", "sst", "chs", "chs2", "cqs", "cqs2", "cpm", "znt", "ust",
      "pblh", "zol", "mol", "psim", "psih", "qgh", "hfx", "qfx", "lh", "flhc",
      "flqc", "qsfc", "rmol", "gz1oz0", "wspd", "br", "ch", "ustm", "ck",
      "cka", "cd", "cda", "u10", "v10", "th2", "t2", "q2", "tsk_sea",
      "qgh_sea", "chs2_sea", "chs_sea", "cpm_sea", "cqs2_sea", "flhc_sea",
      "flqc_sea", "hfx_sea", "lh_sea", "qfx_sea", "qsfc_sea", "znt_sea")
HEADER = ("nx", "ny", "mode", "itimestep", "isfflx", "spp_pbl", "land",
          "water", "shal", "flag_lsm", "restart", "tice2tsk_if2cold")

#: Outputs WRF defines after each call site.  TSK_SEA is left out: ArWen
#: keeps the SST carrier in f["tsk_sea"] (compared as ``sst``), where WRF's
#: TSK_SEA takes TSK at open cells; RUC's seam reads it only at ice cells.
COMPARED_SINGLE = ("chs", "chs2", "cqs", "cqs2", "cpm", "znt", "ust", "pblh",
                   "zol", "mol", "psim", "psih", "qgh", "hfx", "qfx", "lh",
                   "flhc", "flqc", "qsfc", "rmol", "gz1oz0", "wspd", "br",
                   "ch", "ustm", "ck", "cka", "cd", "cda", "u10", "v10",
                   "th2", "t2", "q2")
COMPARED_WRAPPER = COMPARED_SINGLE + (
    "sst", "qgh_sea", "chs2_sea", "chs_sea", "cpm_sea", "cqs2_sea",
    "flhc_sea", "flqc_sea", "hfx_sea", "lh_sea", "qfx_sea", "qsfc_sea",
    "znt_sea")


def write_native_input(path: Path, cap) -> None:
    hdr = np.asarray([int(cap[k]) for k in HEADER], dtype="<i4")
    with open(path, "wb") as fh:
        fh.write(hdr.tobytes())
        fh.write(np.asarray([cap["xice_threshold"]], dtype="<f4").tobytes())
        for name in N3:   # (2, ny, nx) C order == Fortran (nx, 2, ny)
            a = np.ascontiguousarray(np.asarray(cap[name], dtype="<f4")
                                     .transpose(1, 0, 2))
            fh.write(a.tobytes())
        for name in N2:   # (ny, nx) C order == Fortran (nx, ny)
            fh.write(np.ascontiguousarray(cap[name], dtype="<f4").tobytes())


def read_native_output(path: Path, ny: int, nx: int) -> dict:
    raw = np.fromfile(path, dtype="<f4").reshape(len(N2), ny, nx)
    return {name: raw[i] for i, name in enumerate(N2)}


def run_native(exe: str, cap) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        write_native_input(fin, cap)
        subprocess.run([exe, str(fin), str(fout)], check=True)
        return read_native_output(fout, int(cap["ny"]), int(cap["nx"]))


def run_gpu(cap) -> dict:
    import cupy as cp
    from woof.core.physics import PhysicsDriver
    f = {}
    for name in N2:
        if name in ("psfc", "sst", "dx"):
            continue
        f[name] = cp.ascontiguousarray(cp.asarray(cap[name], dtype=cp.float32))
    f["tsk_sea"] = cp.ascontiguousarray(cp.asarray(cap["sst"], dtype=cp.float32))
    from woof.core.mynnsfc_b66187f import SEAICE_OWN
    for name in SEAICE_OWN:
        key = f"{name}_sea"
        if key not in f:
            f[key] = cp.zeros_like(f["xland"])
    atm = {name: cp.ascontiguousarray(cp.asarray(cap[name], dtype=cp.float32))
           for name in N3 if name != "pattern_spp_pbl"}
    atm["p_interface"] = cp.ascontiguousarray(
        cp.asarray(cap["psfc"], dtype=cp.float32)[None])
    mode = int(cap["mode"])
    cfg = SimpleNamespace(
        dx=float(np.asarray(cap["dx"]).flat[0]),
        fractional_seaice=mode, sf_surface_physics=int(cap["flag_lsm"]),
        isfflx=int(cap["isfflx"]), spp_pbl=int(cap["spp_pbl"]),
        sf_mynn_sfcflux_land=int(cap["land"]),
        sf_mynn_sfcflux_water=int(cap["water"]),
        shalwater_z0=int(cap["shal"]))
    owner = SimpleNamespace(
        fields=f, spp_patterns={"pbl": cp.asarray(cap["pattern_spp_pbl"],
                                                  dtype=cp.float32)})
    thr = float(cap["xice_threshold"])
    ice = (f["xice"] >= np.float32(thr)) & (f["xice"] <= np.float32(1.0))
    PhysicsDriver._run_mynnsfc_b66187f(
        owner, atm, cfg, itimestep=int(cap["itimestep"]),
        ice_params=SimpleNamespace(xice_threshold=thr), ice_component=ice,
        carriers=(mode == 1))
    out = {name: cp.asnumpy(f[name]) for name in f}
    out["cqs"] = cp.asnumpy(owner._mynnsfc_b66187f.cqs)
    out["sst"] = cp.asnumpy(f["tsk_sea"])
    return out


def compare(cap, native: dict, gpu: dict) -> dict:
    names = COMPARED_WRAPPER if int(cap["mode"]) == 1 else COMPARED_SINGLE
    report, total, diff = {}, 0, 0
    for name in names:
        a = np.ascontiguousarray(native[name], dtype="<f4").view("<u4")
        b = np.ascontiguousarray(gpu[name], dtype="<f4").view("<u4")
        nd = int(np.count_nonzero(a != b))
        total += a.size
        diff += nd
        entry = {"words": int(a.size), "different": nd}
        if nd:
            fa = native[name].astype(np.float64)
            fb = gpu[name].astype(np.float64)
            k = np.flatnonzero(a.ravel() != b.ravel())
            ulp = np.abs(a.astype(np.int64) - b.astype(np.int64)).ravel()[k]
            entry.update(max_ulp=int(ulp.max()), first=int(k[0]),
                         native=float(fa.ravel()[k[0]]),
                         gpu=float(fb.ravel()[k[0]]))
        report[name] = entry
    return {"words": total, "different": diff, "fields": report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("captures", nargs="+")
    c.add_argument("--native", required=True)
    c.add_argument("--json")
    args = ap.parse_args(argv)
    results = {}
    words = different = 0
    for path in args.captures:
        cap = dict(np.load(path))
        native = run_native(args.native, cap)
        gpu = run_gpu(cap)
        r = compare(cap, native, gpu)
        results[Path(path).name] = r
        words += r["words"]
        different += r["different"]
        bad = {k: v for k, v in r["fields"].items() if v["different"]}
        print(f"{Path(path).name}: {r['words']} words, {r['different']} "
              f"different {json.dumps(bad) if bad else ''}")
    print(f"TOTAL {words} words, {different} different")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"words": words, "different": different, "captures": results},
            indent=1))
    return 0 if different == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
