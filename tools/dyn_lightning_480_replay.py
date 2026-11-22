"""Module replay: WRF v4.8.0 dynamic lightning, GPU vs the native strict referee.

Real columns: the Thompson mp8 wrapper columns bound to the strict referee
(tests/data/oracles/thompson_bound239/stock.npz, 42 HRRR-case columns, the
native post-scheme state) and the NSSL 2-moment column oracle
(tests/data/oracles/nssl2_column/nssl2-column-oracle-v2.npz, 96 columns with
hail). The same float32 bytes go to both sides. Each set is laid out as a
2-D tile so calc_dyn_pe's column-to-column carry is exercised in WRF's DO j /
DO i order, and the three routines run for N consecutive steps so charge
accumulates and flashes discharge.

Native drivers (built by build_native.sh in localhost/wrfref2404:1):
  pristine: the referee build's own module objects (WRF-4.8.0-serial-pristine);
  patched : calc_lpi_new with three labelled lines that define the kte slots.
Pristine runs repeat with different heap prefills; words that change with the
prefill are the native-undefined witness set.

Usage (a CUDA host with CuPy):
  GPUWM_WRF_EXACT=1 python tools/dyn_lightning_480_replay.py --replay-dir DIR --steps 720
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / "tests" / "data" / "oracles"
PARAMS = dict(coul_pos=0.25e-4, coul_neg=0.25e-4, coul_neu=0.25e-4,
              j_pos=5.e9, j_neg=1.e9, j_neu=1.e9)
PATTERNS = (0.0, 1.0e-3, -5.0)
OUT_NAMES = ("lpi2d", "lpi3d", "light_pe", "light_ne", "light_neu", "lpos", "lneg", "lneu")


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def thompson_set():
    z = np.load(ORACLES / "thompson_bound239" / "stock.npz")
    th, pii, p, qv = (_f32(z[k]) for k in ("wrf_th", "wrapper_pii", "wrapper_p", "wrf_qv"))
    tk = th * pii
    rho = _f32(p / (np.float32(287.04) * tk * (np.float32(1.) + np.float32(0.61) * qv)))
    cols = dict(th=th, pii=pii, p=p, rho=rho, dz8w=_f32(z["wrapper_dz"]), w=_f32(z["col_w"]),
                qc=_f32(z["wrf_qc"]), qr=_f32(z["wrf_qr"]), qi=_f32(z["wrf_qi"]),
                qs=_f32(z["wrf_qs"]), qg=_f32(z["wrf_qg"]))
    cols["qh"] = np.zeros_like(cols["qc"])
    return "thompson_bound239", cols, (6, 7)


def nssl_set():
    z = np.load(ORACLES / "nssl2_column" / "nssl2-column-oracle-v2.npz")
    cols = dict(th=_f32(z["in_theta"]), pii=_f32(z["in_exner"]), p=_f32(z["in_pressure"]),
                rho=_f32(z["in_rho"]), dz8w=_f32(z["in_dz"]), w=_f32(z["in_w"]),
                qc=_f32(z["in_qc"]), qr=_f32(z["in_qr"]), qi=_f32(z["in_qi"]),
                qs=_f32(z["in_qs"]), qg=_f32(z["in_qg"]), qh=_f32(z["in_qh"]))
    return "nssl2_column", cols, (8, 12)


def to_grid(cols, shape):
    """[col, k] -> [k, ny, nx] (tile order j-major, i-minor)."""
    ny, nx = shape
    out = {}
    for name, a in cols.items():
        out[name] = np.ascontiguousarray(a.T.reshape(a.shape[1], ny, nx))
    return out


def write_native_input(path, g, nsteps, nsave, dx, dy, dt, pattern):
    nz, ny, nx = g["th"].shape
    kme = nz + 1
    def ikj(a, levels):
        full = np.zeros((kme, ny, nx), np.float32)
        full[:levels] = a
        return np.ascontiguousarray(full.transpose(1, 0, 2))      # (j,k,i) C = (i,k,j) F
    with open(path, "wb") as f:
        f.write(np.array([nx, ny, nz, nsteps, nsave], np.int32).tobytes())
        f.write(np.array([dx, dy, dt, PARAMS["coul_pos"], PARAMS["coul_neg"], PARAMS["coul_neu"],
                          PARAMS["j_pos"], PARAMS["j_neg"], PARAMS["j_neu"], pattern], np.float32).tobytes())
        for name in ("th", "pii", "p", "rho", "dz8w"):
            f.write(ikj(g[name], nz).tobytes())
        f.write(ikj(g["w"], nz + 1).tobytes())
        for name in ("qc", "qr", "qi", "qs", "qg", "qh"):
            f.write(ikj(g[name], nz).tobytes())
        zero3 = np.zeros((ny, kme, nx), np.float32)
        for _ in range(3):
            f.write(zero3.tobytes())
        zero2 = np.zeros((ny, nx), np.float32)
        for _ in range(3):
            f.write(zero2.tobytes())


def read_native_output(path, nz, ny, nx, nsave):
    raw = np.fromfile(path, dtype=np.float32)
    per = ny * nx * (4 + 4 * nz)
    if raw.size != per * nsave:
        raise RuntimeError(f"native output {path} has {raw.size} words, expected {per * nsave}")
    steps = []
    for s in range(nsave):
        r = raw[s * per:(s + 1) * per]
        o, pos = {}, 0
        for name in OUT_NAMES:
            levels = nz if name in ("lpi3d", "light_pe", "light_ne", "light_neu") else 1
            n = ny * nx * levels
            a = r[pos:pos + n]; pos += n
            # Fortran (i,k,j) -> [k, j, i]
            o[name] = (a.reshape(ny, levels, nx).transpose(1, 0, 2) if levels > 1
                       else a.reshape(ny, nx))
        steps.append(o)
    return steps


def run_native(replay_dir, kind, inp, out):
    exe = f"/r/native/{kind}/dynltng_{kind}"
    cmd = ["podman", "run", "--rm", "-v", f"{replay_dir}:/r", "-w", "/r",
           "localhost/wrfref2404:1", exe, "/r/" + inp.name, "/r/" + out.name]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def run_gpu(g, nsteps, dx, dy, dt, timing_steps=0):
    import cupy as cp
    from woof.core.dyn_lightning import launch_dyn_lightning
    nz, ny, nx = g["th"].shape
    fields = {k: cp.asarray(v) for k, v in g.items() if k != "p"}
    out = {k: cp.zeros((nz, ny, nx), cp.float32) for k in ("light_pe", "light_ne", "light_neu", "lpi3d")}
    out.update({k: cp.zeros((ny, nx), cp.float32) for k in ("lpos", "lneg", "lneu", "lpi2d")})
    steps = []
    for _ in range(nsteps):
        launch_dyn_lightning(fields, out, dx=dx, dy=dy, dt=dt, **PARAMS)
        steps.append({k: cp.asnumpy(v) for k, v in out.items()})
    return steps


def compare(a_steps, b_steps, mask_steps=None):
    total = differ = 0
    per = {}
    first = None
    for s, (a, b) in enumerate(zip(a_steps, b_steps)):
        for name in OUT_NAMES:
            x = a[name].view(np.uint32); y = b[name].view(np.uint32)
            keep = np.ones(x.shape, bool) if mask_steps is None else ~mask_steps[s][name]
            d = (x != y) & keep
            nd = int(d.sum()); nk = int(keep.sum())
            total += nk; differ += nd
            row = per.setdefault(name, [0, 0]); row[0] += nk; row[1] += nd
            if nd and first is None:
                first = {"step": s + 1, "field": name, "index": [int(v) for v in np.argwhere(d)[0]]}
    return {"words": total, "different": differ, "per_field": per, "first_difference": first}


def witness(pristine_runs):
    """Words whose value changes with the heap prefill."""
    masks = []
    base = pristine_runs[0]
    for s in range(len(base)):
        m = {}
        for name in OUT_NAMES:
            bits = base[s][name].view(np.uint32)
            mm = np.zeros(bits.shape, bool)
            for other in pristine_runs[1:]:
                mm |= other[s][name].view(np.uint32) != bits
            m[name] = mm
        masks.append(m)
    return masks


def coverage(steps):
    last = steps[-1]
    return {
        "columns": int(last["lpi2d"].size),
        "columns_lpi2d_positive": int((last["lpi2d"] > 0).sum()),
        "lpi3d_positive_words": int((last["lpi3d"] > 0).sum()),
        "columns_with_positive_flashes": int((last["lpos"] > 0).sum()),
        "columns_with_negative_flashes": int((last["lneg"] > 0).sum()),
        "columns_with_ic_flashes": int((last["lneu"] > 0).sum()),
        "total_flashes": {k: float(last[k].sum()) for k in ("lpos", "lneg", "lneu")},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--dx", type=float, default=3000.)
    ap.add_argument("--dt", type=float, default=15.)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--oracle-dir", type=Path, default=None,
                    help="also store the inputs and the patched native words at a few steps")
    a = ap.parse_args(argv)
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise SystemExit("run with GPUWM_WRF_EXACT=1: the replay is the strict comparison")
    rd = a.replay_dir.resolve()
    receipt = {"schema": "dyn-lightning-480-replay-v1", "steps": a.steps, "dx": a.dx, "dy": a.dx,
               "dt": a.dt, "params": PARAMS, "native_sha256": (rd / "native" / "SHA256SUMS").read_text(),
               "kernel_sha256": hashlib.sha256((ROOT / "woof/core/kernels/dyn_lightning.cu").read_bytes()).hexdigest(),
               "sets": {}}
    for build in (thompson_set, nssl_set):
        name, cols, shape = build()
        g = to_grid(cols, shape)
        nz, ny, nx = g["th"].shape
        row = {"tile": [ny, nx], "nz": nz,
               "input_sha256": hashlib.sha256(b"".join(g[k].tobytes() for k in sorted(g))).hexdigest()}
        runs = {}
        for kind in ("pristine", "patched"):
            for pattern in (PATTERNS if kind == "pristine" else PATTERNS[:2]):
                tag = f"{name}.{kind}.{pattern:g}"
                inp = rd / f"{tag}.in"; out = rd / f"{tag}.out"
                write_native_input(inp, g, a.steps, a.steps, a.dx, a.dx, a.dt, pattern)
                row[f"native_seconds.{kind}.{pattern:g}"] = run_native(rd, kind, inp, out)
                runs[(kind, pattern)] = read_native_output(out, nz, ny, nx, a.steps)
                inp.unlink(); out.unlink()
        gpu = run_gpu(g, a.steps, a.dx, a.dx, a.dt)
        pristine = [runs[("pristine", p)] for p in PATTERNS]
        mask = witness(pristine)
        row["witness_words"] = {n: int(sum(int(m[n].sum()) for m in mask)) for n in OUT_NAMES}
        row["patched_prefill_independent"] = compare(runs[("patched", PATTERNS[0])], runs[("patched", PATTERNS[1])])
        row["gpu_vs_patched"] = compare(gpu, runs[("patched", PATTERNS[0])])
        row["gpu_vs_pristine_defined_words"] = compare(gpu, runs[("pristine", PATTERNS[0])], mask)
        row["patched_vs_pristine_defined_words"] = compare(runs[("patched", PATTERNS[0])], runs[("pristine", PATTERNS[0])], mask)
        row["coverage"] = coverage(gpu)
        if a.oracle_dir is not None:
            a.oracle_dir.mkdir(parents=True, exist_ok=True)
            keep = sorted({1, 2, 60, a.steps // 2, a.steps})
            payload = {k: v for k, v in g.items() if k != "p"}
            payload.update({k: np.float32(v) for k, v in PARAMS.items()})
            payload.update(dx=np.float32(a.dx), dy=np.float32(a.dx), dt=np.float32(a.dt),
                           steps=np.array(keep, np.int32))
            for s in keep:
                for key in OUT_NAMES:
                    payload[f"{key}@{s}"] = runs[("patched", PATTERNS[0])][s - 1][key]
            np.savez_compressed(a.oracle_dir / f"{name}.npz", **payload)
        receipt["sets"][name] = row
        print(name, json.dumps({k: row[k] for k in ("witness_words", "coverage")}), flush=True)
        for k in ("patched_prefill_independent", "gpu_vs_patched", "gpu_vs_pristine_defined_words",
                  "patched_vs_pristine_defined_words"):
            print(" ", k, row[k]["words"], row[k]["different"], row[k]["first_difference"], flush=True)
    a.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
