"""Instrument only the lane-owned native PX module; no arithmetic changes."""
from pathlib import Path
import argparse
import hashlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("wrf_root", type=Path, help="lane-owned copy of the pinned WRF tree")
root = parser.parse_args().wrf_root.resolve()
p=root/'phys/module_sf_pxsfclay.F'
if hashlib.sha256(p.read_bytes()).hexdigest() != "cd1b2095f8e093d4bc423ebbc4a3bfbce58a7d17b34f1f41d770c6061b68d0a4":
 raise SystemExit("PX module differs from pinned WRF v4.6.1/v4.7.1 source")
s=p.read_text()
start=s.index('   SUBROUTINE PXSFCLAY(')
stop=s.index('   END SUBROUTINE PXSFCLAY',start)
part=s[start:stop]
part=part.replace('      INTEGER ::  I,J','      INTEGER ::  I,J, pxunit\n      CHARACTER(LEN=128) :: pxpath\n      LOGICAL :: pxcapture')
inputs=['U1D(its:ite)','V1D(its:ite)','T1D(its:ite)','TH1D(its:ite)','QV1D(its:ite)','P1D(its:ite)','dz8w1d(its:ite)']+[f'{name}(its:ite,j)' for name in ['PSFC','TSK','PBLH','MAVAIL','XLAND']]
state=[f'{name}(its:ite,j)' for name in ['REGIME','HFX','QFX','LH','MOL','RMOL','QSFC','GZ1OZ0','WSPD','BR','PSIM','PSIH','ZNT','ZOL','UST','CPM','CHS2','CQS2','CHS','FLHC','FLQC','QGH','U10','V10']]
constants='CP,G,ROVCP,R,XLV,DX,SVP1,SVP2,SVP3,SVPT0,EP1,EP2,KARMAN'
pre='''        pxcapture = ITIMESTEP <= 3 .AND. J == jts
        IF (pxcapture) THEN
          WRITE(pxpath,'("pxcap-",I5.5,"-",I5.5,"-",I5.5,".bin")') its,J,ITIMESTEP
          OPEN(NEWUNIT=pxunit,FILE=TRIM(pxpath),ACCESS='STREAM', &
               FORM='UNFORMATTED',STATUS='REPLACE',CONVERT='little_endian')
          WRITE(pxunit) 1347965776,ite-its+1,ITIMESTEP,J
          WRITE(pxunit) '''+constants+'\n'+''.join('          WRITE(pxunit) '+x+'\n' for x in inputs+state)+'''        END IF
'''
part=part.replace('        CALL PXSFCLAY1D(',pre+'        CALL PXSFCLAY1D(',1)
post='''        IF (pxcapture) THEN
'''+''.join('          WRITE(pxunit) '+x+'\n' for x in state)+'''          CLOSE(pxunit)
        END IF
'''
# Insert after the complete call, immediately before the j-loop end.
mark='                its,ite, jts,jte, kts,kte                          )\n'
assert part.count(mark)==1
part=part.replace(mark,mark+post)
p.write_text(s[:start]+part+s[stop:])
print(f"instrumented {p}; rebuild the native tree with strict scalar flags before running")
