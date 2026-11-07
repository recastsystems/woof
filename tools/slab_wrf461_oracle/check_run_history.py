"""Check (c) of check_slab.sh: the slab run's own history, read back.

Passes only if the run directory holds wrfout history written by a
``SF_SURFACE_PHYSICS = 1`` run whose last frame is at least ten model
minutes after its first, with five soil layers, and every TSK and TSLB
value of every frame finite and inside 150..350 K.  Prints the per-frame
ranges as its receipt.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def main(run_dir: str) -> int:
    from woof.netcdf_bridge import Dataset

    files = sorted(p for p in Path(run_dir).rglob("wrfout_d01_*")
                   if p.is_file() and p.suffix in ("", ".nc"))
    if not files:
        print(f"no wrfout_d01 history under {run_dir}")
        return 1
    times = []
    ok = True
    for path in files:
        with Dataset(path) as ds:
            scheme = int(getattr(ds, "SF_SURFACE_PHYSICS", -1))
            if float(getattr(ds, "DX", -1)) != 12000. or float(getattr(ds, "DY", -1)) != 12000.:
                print(f"{path.name}: grid spacing is not 12 km")
                ok = False
            if scheme != 1:
                print(f"{path.name}: SF_SURFACE_PHYSICS={scheme}, not 1")
                ok = False
            tslb = np.asarray(ds.variables["TSLB"][:], dtype=np.float64)
            tsk = np.asarray(ds.variables["TSK"][:], dtype=np.float64)
            if tslb.shape[1] != 5:
                print(f"{path.name}: TSLB has {tslb.shape[1]} layers")
                ok = False
            stamps = ["".join(c.decode() for c in row)
                      for row in np.asarray(ds.variables["Times"][:])]
            for k, stamp in enumerate(stamps):
                times.append(datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S"))
                t, s = tsk[k], tslb[k]
                finite = bool(np.isfinite(t).all() and np.isfinite(s).all())
                inside = bool(finite and t.min() > 150 and t.max() < 350
                              and s.min() > 150 and s.max() < 350)
                print(f"{stamp} TSK {np.nanmin(t):.3f}..{np.nanmax(t):.3f} "
                      f"TSLB {np.nanmin(s):.3f}..{np.nanmax(s):.3f} "
                      f"finite={finite}")
                ok &= inside
    span = (max(times) - min(times)).total_seconds()
    print(f"history spans {span:.0f} s over {len(times)} frames")
    ok &= span >= 600.0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
