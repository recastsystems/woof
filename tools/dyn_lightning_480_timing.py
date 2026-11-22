"""Timing of one dynamic-lightning call (the three WRF routines) on a domain
tiled from the 96 real NSSL columns. Device events only; no profiler.

  python tools/dyn_lightning_480_timing.py --nx 600 --ny 600 --calls 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=600)
    ap.add_argument("--ny", type=int, default=600)
    ap.add_argument("--calls", type=int, default=30)
    a = ap.parse_args(argv)
    import cupy as cp
    try:
        from tools.dyn_lightning_480_replay import nssl_set, PARAMS
    except ImportError:  # run as a path from tools/
        from dyn_lightning_480_replay import nssl_set, PARAMS
    from woof.core.dyn_lightning import launch_dyn_lightning
    from woof.wrf_exact import ENABLED
    _, cols, _ = nssl_set()
    ncol = a.nx * a.ny
    pick = np.arange(ncol) % next(iter(cols.values())).shape[0]
    fields = {}
    for k, v in cols.items():
        if k == "p":
            continue
        fields[k] = cp.asarray(np.ascontiguousarray(v[pick].T.reshape(v.shape[1], a.ny, a.nx)))
    nz = fields["th"].shape[0]
    out = {k: cp.zeros((nz, a.ny, a.nx), cp.float32) for k in ("light_pe", "light_ne", "light_neu", "lpi3d")}
    out.update({k: cp.zeros((a.ny, a.nx), cp.float32) for k in ("lpos", "lneg", "lneu", "lpi2d")})
    for _ in range(3):
        launch_dyn_lightning(fields, out, dx=3000., dy=3000., dt=15., **PARAMS)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(a.calls):
        start.record()
        launch_dyn_lightning(fields, out, dx=3000., dy=3000., dt=15., **PARAMS)
        stop.record(); stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))
    t = np.array(times)
    print(json.dumps({"strict": bool(ENABLED), "nx": a.nx, "ny": a.ny, "nz": nz,
                      "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                      "median_ms": float(np.median(t)), "min_ms": float(t.min()),
                      "max_ms": float(t.max()), "calls": a.calls,
                      "ns_per_column": float(np.median(t)) * 1e6 / (a.nx * a.ny)}))


if __name__ == "__main__":
    sys.exit(main())
