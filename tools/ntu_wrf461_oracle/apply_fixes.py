#!/usr/bin/env python3
"""Apply the declared NTU divergences (WRF bugs WOOF fixes) to a copy of
module_mp_ntu.F.  Each edit must match exactly once."""
import sys

FIXES = [
 ("F0 INIT_AEROSOL level loop reads one level past the filled mass levels",
  "            K_PBL = 1\n            DO K = KTS,KTE\n",
  "            K_PBL = 1\n            DO K = KTS,KTE-1\n"),
 ("F1 ICENU: ANDvi read INDvi before INDvi was assigned",
  "      ANDvi = (KCIMIN*NNDvi*INDvi**2.)**THRD\n      INDvi = QNDvi*iAMI0\n",
  "      INDvi = QNDvi*iAMI0\n      ANDvi = (KCIMIN*NNDvi*INDvi**2.)**THRD\n"),
 ("F2 SMALL_DT: graupel sublimation volume read QVDvg after zeroing it",
  "               QSBgv = MIN(QVDvg,0.)\n               QVDvg = 0.\n               VSBgv = MIN(QVDvg/RHOG,0.)\n",
  "               QSBgv = MIN(QVDvg,0.)\n               VSBgv = MIN(QVDvg/RHOG,0.)\n               QVDvg = 0.\n"),
 ("F3 LARGE_DT: GR7 used the sixth moment gamma (copy of GR6)",
  "         GR7   = EXP(GAMLN(AFAR+6.)-GAMLN(AFAR+1.)-6.*LOG(LAMR))\n",
  "         GR7   = EXP(GAMLN(AFAR+7.)-GAMLN(AFAR+1.)-6.*LOG(LAMR))\n"),
 ("F4 LARGE_DT: rain second-moment fall speed multiplied by 1/4 instead of raising to 1/4",
  "            VTAR = VTQR**(3./4.)*VTNR*(1./4.)\n         ELSEIF (LIQ_VTR.EQ.2) THEN",
  "            VTAR = VTQR**(3./4.)*VTNR**(1./4.)\n         ELSEIF (LIQ_VTR.EQ.2) THEN"),
 ("F5 LARGE_DT: rain-graupel collection read DNGRM before assigning it",
  "            RHOGW = (QG1D+QR1D)/(QG1D/RHOG+QR1D/RHOW+ISMALL)\n            RATIO = (RHOGW/DNGRM)**THRD\n            STOKE = (RHOW*ABS(VTQ0)*MVDR**2.)/(9.*MUA*MVDG)\n",
  "            RHOGW = (QG1D+QR1D)/(QG1D/RHOG+QR1D/RHOW+ISMALL)\n            STOKE = (RHOW*ABS(VTQ0)*MVDR**2.)/(9.*MUA*MVDG)\n"),
 ("F5b (move) ",
  "               DNGRM = 3.E2*(-5.E5*MVDR*0.6*VTQRG/TC1D)**0.44\n            ENDIF\n            DNGRM = MIN(MAX(DNGRM,RHOIMIN),RHOG0)\n",
  "               DNGRM = 3.E2*(-5.E5*MVDR*0.6*VTQRG/TC1D)**0.44\n            ENDIF\n            DNGRM = MIN(MAX(DNGRM,RHOIMIN),RHOG0)\n            RATIO = (RHOGW/DNGRM)**THRD\n"),
 ("F6 LARGE_DT: graupel melting second moment used rain gammas for collected cloud",
  "                  RMcga = (MVDX**2.-MVDG**2.)*NRMC1*VTQGC*NG1D*(GG3+   &\n                          2.*GG2*GR2+GR3)\n",
  "                  RMcga = (MVDX**2.-MVDG**2.)*NRMC1*VTQGC*NG1D*(GG3+   &\n                          2.*GG2*GC2+GC3)\n"),
 ("F6b LARGE_DT: hail melting second moment used rain gammas for collected cloud",
  "                  RMcha = (MVDX**2.-MVDH**2.)*NRMC1*VTQHC*NH1D*(GH3+   &\n                          2.*GH2*GR2+GR3)\n",
  "                  RMcha = (MVDX**2.-MVDH**2.)*NRMC1*VTQHC*NH1D*(GH3+   &\n                          2.*GH2*GC2+GC3)\n"),
]

def main(src, dst):
    s = open(src, encoding="latin-1").read()
    for name, old, new in FIXES:
        n = s.count(old)
        if n != 1:
            sys.exit(f"{name}: matched {n} times")
        s = s.replace(old, new)
    open(dst, "w", encoding="latin-1").write(s)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
