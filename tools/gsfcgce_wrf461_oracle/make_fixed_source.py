#!/usr/bin/env python3
"""Write the FIX arm of the Goddard GCE oracle: WRF's preprocessed source
with exactly the two undefined-read repairs woof declares, and nothing else.

    python make_fixed_source.py <module_mp_gsfcgce.f90> <out.f90>

Each repair is an anchored replacement that must match exactly once, so a
source that is not the pinned v4.6.1 text fails loudly instead of yielding
a half-patched referee.  The two undefined reads (woof/core/gsfcgce_constants.py
``DECLARED_DIVERGENCES`` carries the same list):

1. ``gsfcgce_2ice = 3`` (warm rain only): ``saticel_s`` evaporates rain with
   the air viscosity ``scv``, thermal conductivity ``tca`` and vapour
   diffusivity ``dwv`` that only the ice branch computes; in the warm
   branch they are automatic arrays read uninitialised.  Fixed: the warm
   branch computes them with the ice branch's four statements, at the same
   point (after the rain slope), from the same level-entry temperature.
2. ``gsfcgce_2ice = 1`` (cloud ice + snow, no graupel): ``refl10cm_gsfc``
   is handed a ``qg1d`` column that this branch never fills.  Fixed: the
   branch zeroes it, as the warm-only branch already does.
"""
from __future__ import annotations

import sys

REPAIRS = (
    ("warm-transport-coefficients",
     "                pracw(i,j)=0.\n                praut(i,j)=0.0\n",
     "            y1(i,j)=c149*tair(i,j)**1.5/(tair(i,j)+120.)\n"
     "            dwv(i,j)=dwvp*tair(i,j)**1.81\n"
     "            tca(i,j)=c141*y1(i,j)\n"
     "            scv(i,j)=1./((rr0*y1(i,j))**.1666667*dwv(i,j)**.3333333)\n"
     "                pracw(i,j)=0.\n                praut(i,j)=0.0\n"),
    ("refl-2ice-snow-qg1d",
     "               elseif (ice2.eq.1) then\n                  DO K=kts,kte\n"
     "                     qs1d(k)=qswrf(i,k,j)\n",
     "               elseif (ice2.eq.1) then\n                  DO K=kts,kte\n"
     "                     qs1d(k)=qswrf(i,k,j)\n"
     "                     qg1d(k)=0.\n"),
)


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src, encoding="ascii").read()
    for name, old, new in REPAIRS:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"repair {name}: anchor matches {count} times")
        text = text.replace(old, new)
    open(dst, "w", encoding="ascii").write(text)
    print(f"wrote {dst}: {len(REPAIRS)} repairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
