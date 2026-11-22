# SPDX-License-Identifier: Apache-2.0
"""Gradients of two measures of your own on the 12 km example, from Python (a 32 GB compute capability 12.0 card).

    python own_measure.py OUT_DIR

Each measure is a mean over a region and levels at the end of the 15-minute window; the gradient is with respect to
the whole start state. Each gradient lands in OUT_DIR/g0001, g0002 (sensitivity.nc, receipt.json), exactly as the
`woof hex sensitivity` command writes it. The script prints where each gradient to the measure's own start field
(theta for theta_m, water vapour for qv) is largest, with its latitude and longitude, and whether its answer check
passed, and exits 1 if either did not (2 if the command refused, in one sentence).
"""
import sys

import numpy as np

from woof.hex.ad_session import SensitivitySession

ROW = "hrrrv5-q12.96.2890.n40.60w91.30"
MEASURES = [
    ("theta_m", {"circle": "40.6,-91.3,150", "levels": "1-10"}),   # low-level temperature in a 150 km circle
    ("qv", {"circle": "40.6,-91.3,100", "levels": "1-7"}),          # near-ground vapour in a 100 km circle
]


def main(out: str) -> int:
    ok = True
    with SensitivitySession(out, row=ROW, steps=15) as s:
        for metric, where in MEASURES:
            g = s.gradient(metric, **where)
            name = "dJ_dqv" if metric == "qv" else "dJ_dtheta"   # the gradient to the measure's own start field
            d = np.asarray(g["fields"][name])                   # [levels, cells]
            lev, cell = np.unravel_index(np.argmax(np.abs(d)), d.shape)
            print(f"{metric} {where}: valid {g['valid']}, passed {g['passed']}, {g['seconds']:.1f} s; "
                  f"largest |{name.replace('_', '/', 1)}| {abs(d[lev, cell]):.3e} at level {lev + 1}, "
                  f"{g['latCell'][cell]:.2f}N {g['lonCell'][cell]:.2f}E (cell index {cell}); folder {g['out']}")
            ok = ok and bool(g["passed"])
    return 0 if ok else 1


if __name__ == "__main__":
    from woof.hex.ad_doors import AdDoorError
    try:
        sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "own-measure"))
    except AdDoorError as refused:   # a refusal is one sentence and exit status 2, as the commands say it
        print(f"own_measure: {refused}", file=sys.stderr)
        sys.exit(2)
