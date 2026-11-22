#!/usr/bin/env python3
"""Apply the committed both-sides reduction to ONE side's npz.

This defines nothing.  It imports `reduce_moist` from `same_instrument_moist`
-- the routine that was written and committed before any moist run existed --
and runs it on a single `<prefix>_moist_profiles.npz`, so the WRF arm's
resolved fractions are computed by exactly the routine the WOOF-ArWen comparison
will use, rather than by a second implementation that could quietly differ.

Written after the campaign data existed, and declared as such in
INSTRUMENT-HISTORY.md.  It is an entry point, not a metric: every number it
prints comes out of `reduce_moist` unchanged, and it cuts no band.

Usage: reduce_moist_one.py <prefix>_moist_profiles.npz [--window-min 30]
                           [--json out.json] [--label NAME]
"""
import sys
import json
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from same_instrument_moist import reduce_moist, REQUIRED  # noqa: E402


def main():
    path = sys.argv[1]
    win = 30.0
    if "--window-min" in sys.argv:
        win = float(sys.argv[sys.argv.index("--window-min") + 1])
    label = os.path.basename(path)
    if "--label" in sys.argv:
        label = sys.argv[sys.argv.index("--label") + 1]

    d = np.load(path)
    missing = [k for k in REQUIRED if k not in d.files]
    if missing:
        raise SystemExit("%s is missing %s" % (path, missing))

    r = reduce_moist(d["z_mass"], d["t_seconds"], d["wthv_res"], d["wthv_sgs"],
                     d["wqv_res"], d["wqv_sgs"], d["qv"], d["qc"], d["qr"],
                     d["cloud_frac"], d["sat_frac"], d["n2_moist_frac"],
                     d["lwp"], win * 60.0)
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    out["label"] = label
    out["npz"] = os.path.abspath(path)
    out["window_min"] = win
    for k in sorted(out):
        print("%-30s %s" % (k, out[k]))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w",
                  newline="\n") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
