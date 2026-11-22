#!/usr/bin/env python3
"""Write the LABELLED-PATCHED WRF v4.8.0 Shin-Hong wrapper (TKE copy-in).

    python patch_tke_copyin.py PREPROCESSED_WRAPPER.f90 OUT.f90

WRF v4.8.0 phys/module_bl_shinhong.F declares ``tke`` intent(out) and never
copies it into ``tke_hv`` before ``bl_shinhong_run``, which reads it
(physics_mmm/bl_shinhong.F90 ``q2x(i,k) = 2.*tke(i,k)``): the TKE chain
starts every call from uninitialised stack memory, so TKE_PBL and EL_PBL
are undefined words (tendencies do not read them).  The 4.6.1/4.7.1 wrapper
carried TKE_PBL intent(inout).  This patch restores that: tke becomes
intent(inout) and is copied in beside the other column inputs.  Nothing
else changes; the referee built from it is always labelled patched.
"""
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
s = src.read_text()
decl = ("             intent(out  )   ::                                           tke, &\n"
        "                                                                           el\n")
assert s.count(decl) == 1
s = s.replace(decl,
              "             intent(inout)   ::                                           tke\n"
              "   real(kind=kind_phys),     dimension( ims:ime, kms:kme, jms:jme )          , &\n"
              "             intent(out  )   ::                                            el\n")
copy = "            rthraten_hv(i,k) = rthraten(i,k,j)\n"
assert s.count(copy) == 1
s = s.replace(copy, copy + "            tke_hv(i,k) = tke(i,k,j)\n")
out.write_text(s)
print(f"patched {src} -> {out}")
