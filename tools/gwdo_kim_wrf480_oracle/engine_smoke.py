#!/usr/bin/env python3
"""Step the engine with orographic drag on and write every carrier's words.

    python engine_smoke.py OUT.npz --gwd-opt 1 [--gwdo-version wrf_471]
                           [--steps N]

Run from a tree's root (it imports that tree's woof and tilestream).  It
builds the tilestream harness's 64 x 64 x 12 real-terrain Lambert domain
(3 km, YSU, revised MM5 surface layer, WSM6, gwd_opt as given), hands the
physics a static set with spatially varying orographic statistics, steps
N times and saves the physics carrier inventory.  Two trees' files are
compared word for word by ``--compare A.npz B.npz``.  Without
``--gwdo-version`` the option is not passed at all, so the same command
runs on a tree that predates it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))


def _static(cfg, terrain, names):
    rng = np.random.default_rng(20261109)
    shape = (cfg.ny, cfg.nx)
    hgt = np.asarray(terrain, np.float64)
    out = {"HGT_M": terrain}
    var = 40.0 + 400.0 * (hgt - hgt.min()) / max(np.ptp(hgt), 1.0)
    for name in names:
        if name == "VAR" or name.startswith("VAR"):
            v = var
        elif name.startswith("CON"):
            v = rng.uniform(0.5, 8.0, shape)
        elif name.startswith("OA"):
            v = rng.uniform(-0.8, 0.8, shape)
        elif name.startswith("OL"):
            v = rng.uniform(0.05, 0.9, shape)
        elif name == "ELVMAX":
            v = hgt + rng.uniform(1.0, 3.5, shape) * var
        else:
            raise SystemExit(f"no smoke value for {name}")
        out[name] = np.ascontiguousarray(v, np.float32)
    return out


def run(args) -> None:
    import cupy as cp
    from tilestream import harness, physics_inventory

    over = dict(dt=6.0, dx=3000.0, dy=3000.0, time_step_sound=4,
                terrain_opt=1, map_proj=1, moist=True, mp_physics=6,
                ra_physics=0, cu_physics=0, bl_pbl_physics=1,
                sf_sfclay_physics=1, sf_surface_physics=0,
                gwd_opt=args.gwd_opt)
    if args.gwdo_version:
        over["gwdo_version"] = args.gwdo_version
    cfg = harness.make_config(64, 64, 12, **over)
    geo = harness.make_geography(cfg, terrain=True)
    try:
        from woof.static.orographic import drag_static_fields
        names = drag_static_fields(0, args.gwd_opt,
                                   getattr(cfg, "gwdo_version", "wrf_471"))
    except ImportError:
        from woof.static.orographic import required_static_fields
        names = required_static_fields(0, args.gwd_opt)
    static = _static(cfg, geo.terrain, names)
    state, _ = harness.make_physics_state(cfg, 4242, geography=geo,
                                          terrain_drag_static=static)
    harness.run_steps(state, cfg, args.steps)
    cp.cuda.runtime.deviceSynchronize()
    inv = physics_inventory.carrier_inventory(state)
    np.savez(args.out, **{k: cp.asnumpy(v) for k, v in inv.items()})
    print(f"{args.out}: {len(inv)} carriers after {args.steps} steps, "
          f"static {sorted(static)}")


def compare(a, b) -> int:
    with np.load(a) as fa, np.load(b) as fb:
        keys = sorted(set(fa.files) | set(fb.files))
        words = diff = 0
        bad = []
        for k in keys:
            if k not in fa.files or k not in fb.files:
                bad.append((k, "missing"))
                continue
            x, y = fa[k], fb[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                bad.append((k, "shape/dtype"))
                continue
            n = int((x.view(np.uint8).reshape(-1, x.itemsize)
                     != y.view(np.uint8).reshape(-1, y.itemsize))
                    .any(axis=1).sum())
            words += x.size
            diff += n
            if n:
                bad.append((k, n))
    print(f"{a} vs {b}: {len(keys)} carriers, {words} words compared, "
          f"{diff} different; {bad[:12]}")
    return 0 if diff == 0 and not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--gwd-opt", type=int, default=1)
    ap.add_argument("--gwdo-version")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
