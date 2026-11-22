#!/usr/bin/env python3
"""Refinement of the resolved/subgrid partition: WRF em_les beside WOOF-ArWen.

WRF points are measured in this lane (200 m and 100 m). WOOF-ArWen points are its
published figures at 100 m and 50 m (docs/public/LES.md:89-97). The two
models' pairs are at different spacings, so this plots what each measured and
does not interpolate between them.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_COL = "#2b6cb0"
W_COL = "#c05621"

# measured here
WRF = {
    "km_opt=2 resolved TKE fraction":  {200: 0.8388, 100: 0.8927},
    "km_opt=2 resolved flux fraction": {200: 0.7595, 100: 0.8615},
    "km_opt=3 resolved flux fraction": {200: 0.7411, 100: 0.8340},
}
# ArWen published
ARWEN = {
    "km_opt=2 resolved TKE fraction":  {100: 0.894, 50: 0.932},
    "km_opt=2 resolved flux fraction": {100: 0.838, 50: 0.896},
    "km_opt=3 resolved flux fraction": {100: 0.844, 50: 0.904},
}


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    keys = list(WRF)
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)
    for i, k in enumerate(keys):
        a = ax[i]
        wx = sorted(WRF[k])
        wy = [WRF[k][x] for x in wx]
        axx = sorted(ARWEN[k])
        ayy = [ARWEN[k][x] for x in axx]
        a.plot(wx, wy, "o-", color=W_COL, lw=2.0, ms=7,
               label="WRF em_les (this lane)")
        a.plot(axx, ayy, "s--", color=A_COL, lw=2.0, ms=7,
               label="WOOF-ArWen (published)")
        for x, y in zip(wx, wy):
            a.annotate("%.4f" % y, (x, y), textcoords="offset points",
                       xytext=(6, -12), fontsize=8, color=W_COL)
        for x, y in zip(axx, ayy):
            a.annotate("%.3f" % y, (x, y), textcoords="offset points",
                       xytext=(6, 6), fontsize=8, color=A_COL)
        a.set_xscale("log")
        a.set_xticks([50, 100, 200])
        a.set_xticklabels(["50", "100", "200"])
        a.invert_xaxis()
        a.set_xlabel("dx (m): refining to the right")
        a.set_title(k, fontsize=10)
        a.grid(alpha=0.25, which="both", lw=0.6)
        if i == 0:
            a.set_ylabel("resolved fraction")
            a.legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Resolved fraction moves toward resolved as the grid refines, "
                 "in both implementations\n"
                 "(different dx pairs; nothing is interpolated between the "
                 "measured points)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = os.path.join(outdir, "refinement_partition.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    main()
