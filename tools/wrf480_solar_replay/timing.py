"""Radiation-call solar geometry cost, calendar plus kernel, on one card.

Times what an RRTMG radiation call pays for its sun on the A071 grid
(151 x 151): the calendar (woof.core.swint.calendar_scalars) and
woof.core.solar.solar_geometry_device, synchronised, warm.  Run it once
from each tree on the same card to compare.
"""
import json
import sys
import time
from datetime import datetime

import numpy as np


def main():
    import cupy as cp
    from woof.core.swint import calendar_scalars
    from woof.core.solar import solar_geometry_device
    F = np.float32
    j, i = np.mgrid[0:151, 0:151]
    lat = cp.asarray((F(41.0) + (j - 75).astype(F) * F(0.027)).ravel())
    lon = cp.asarray((F(-93.5) + (i - 75).astype(F) * F(0.036)).ravel())
    start = datetime(2024, 5, 25, 18)
    out = {}
    for gen in ("wrf_480", "wrf_461"):
        def call(k):
            try:
                jul, xt, gmt = calendar_scalars(start, 180.0 * k, generation=gen)
            except TypeError:  # the earlier tree's signature
                jul, xt, gmt = calendar_scalars(start, 180.0 * k)
            ds, cz, ha = solar_geometry_device(jul, F(xt + F(3.0) * F(0.5)), gmt, lat, lon,
                                               generation=gen)
            return cz
        for k in range(20):
            call(k)
        cp.cuda.Device().synchronize()
        reps = []
        for _ in range(5):
            t = time.perf_counter()
            for k in range(400):
                call(k)
            cp.cuda.Device().synchronize()
            reps.append((time.perf_counter() - t) / 400 * 1e6)
        out[gen] = {"median_us_per_call": float(np.median(reps)), "reps_us": reps}
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
