"""The WRF v4.8.0 kim_tofd oracle's synthetic columns.

Every column of the 4.6.1 oracle (tests/data/oracles/sfclayrev/
sfclayrev-inputs.hex: 110 stress cases from convective land to hurricane
ocean) appears twice, each time with a sub-grid orographic standard deviation
``varf`` from two cycles that straddle every TOFD branch:

* 0, exactly 5 (varf_min: ``varf > varf_min`` is false, no drag), the next
  float above 5 (drag on), 37.5 m;
* 180 m, 620.23 m (the largest VAR in the kept 2019-11-26 CONUS wrfinput),
  2500 m, 25000 m (zf = varf*tofd_factor exceeds za, so the min(.., za)
  clamp acts).

Case id = 1000*copy + 4.6.1 case id.  ``write(directory)`` writes
``sfclayrev480-inputs.hex``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "tests" / "data" / "oracles" / "sfclayrev" / "sfclayrev-inputs.hex"
CYCLES = (
    (np.float32(0.0), np.float32(5.0), np.nextafter(np.float32(5.0), np.float32(6.0)),
     np.float32(37.5)),
    (np.float32(180.0), np.float32(620.22998046875), np.float32(2500.0), np.float32(25000.0)),
)


def rows():
    out = []
    for line in SOURCE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        case = int(tok[0])
        for copy, cycle in enumerate(CYCLES, start=1):
            varf = cycle[case % len(cycle)]
            word = f"{int(np.float32(varf).view(np.uint32)):08X}"
            out.append(f"{1000 * copy + case} " + " ".join(tok[1:]) + " " + word)
    return out


def write(directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sfclayrev480-inputs.hex"
    header = ("# case u v t qv p dz8w psfc tsk pblh mavail xland lakemask dx "
              "znt ust ustm mol hfx qfx qsfc zol varf (float32 hex)")
    path.write_text("\n".join([header, *rows()]) + "\n")
    return path


if __name__ == "__main__":
    import sys
    print(write(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE))
