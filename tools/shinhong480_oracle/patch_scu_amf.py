#!/usr/bin/env python3
"""Write the second LABELLED-PATCHED referee object source (scu arm only).

    python patch_scu_amf.py PREPROCESSED_bl_shinhong.tmp.f90 OUT.f90

With shinhong_scu_mixing on, the stratocumulus-top search can make pblflg
true in a column whose surface buoyancy flux is not positive; bl_shinhong.F90
(:1189-1196) then reads amf2/bmf2, which only the sflux > 0 branch assigns:
the words are left over from an earlier column (undefined on the first).
The patch sets both to 0 before the branch, the value the port uses.  It
changes nothing with scu mixing off (every pblflg column then has
sflux > 0 and assigns both).  The referee built from it is labelled patched.
"""
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
s = src.read_text()
anchor = "       if (pblflg(i).and.sflux(i) > 0.) then\n         amf2      = -snlflux0/(mlfrac-sfcfracn)\n"
assert s.count(anchor) == 1
s = s.replace(anchor, "       amf2 = 0.\n       bmf2 = 0.\n" + anchor)
out.write_text(s)
print(f"patched {src} -> {out}")
