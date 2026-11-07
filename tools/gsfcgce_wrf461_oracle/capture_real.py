#!/usr/bin/env python3
"""Instrument an owned WRF source copy without changing physics arithmetic."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1]).resolve()
path = root / "phys/module_mp_gsfcgce.F"
source = path.read_text()
expected = "43e841f78e97a35e261f7d91d83c6ad54c7bf30a74625e933603d2e8d6126ef1"
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
decl = """
! Lane capture only. No physics operands or control flow are changed.
  INTEGER :: cap_u, cap_j
  LOGICAL :: cap_on
  CHARACTER(LEN=160) :: cap_name
"""
source = source.replace("      INTEGER:: NCALL=0", decl + "\n      INTEGER:: NCALL=0", 1)
capture = """
  cap_on = itimestep .eq. 60
  cap_j = (jts+jte)/2
  IF (cap_on) THEN
    WRITE(cap_name,'(A,I6.6,A,I5.5,A,I5.5)') 'capture-',itimestep,'-',its,'-',jts
    OPEN(newunit=cap_u,file=TRIM(cap_name)//'.meta',status='new')
    WRITE(cap_u,*) ite-its+1,kte-kts+1,dt_in,ihail,ice2,diagflag,do_radar_ref,grav,rhowater,rhosnow
    CLOSE(cap_u)
    OPEN(newunit=cap_u,file=TRIM(cap_name)//'.in',status='new', &
      access='stream',form='unformatted',convert='little_endian')
FIELDS_IN
    CLOSE(cap_u)
    OPEN(newunit=cap_u,file=TRIM(cap_name)//'.pre',status='new', &
      access='stream',form='unformatted',convert='little_endian')
FIELDS_OUT
    CLOSE(cap_u)
  ENDIF
"""
input3 = "th qv ql qr qi qs qg rho pii p z dz8w".split()
input2 = "ht rainnc snownc graupelnc".split()
out3 = "th qv ql qr qi qs qg refl_10cm".split()
out2 = "rainnc rainncv snownc snowncv graupelnc graupelncv sr".split()
def writes(three, two):
    return "\n".join([f"    WRITE(cap_u) {f}(its:ite,kts:kte,cap_j)" for f in three] +
                     [f"    WRITE(cap_u) {f}(its:ite,cap_j)" for f in two])
capture = capture.replace("FIELDS_IN", writes(input3,input2)).replace("FIELDS_OUT",writes(out3,out2))
source = source.replace("      IF (NCALL .EQ. 0) THEN", capture + "\n      IF (NCALL .EQ. 0) THEN", 1)
after = """
  IF (cap_on) THEN
    OPEN(newunit=cap_u,file=TRIM(cap_name)//'.post',status='new', &
      access='stream',form='unformatted',convert='little_endian')
FIELDS_OUT
    CLOSE(cap_u)
  ENDIF
""".replace("FIELDS_OUT", writes(out3,out2))
source = source.replace("   END SUBROUTINE gsfcgce",after + "\n   END SUBROUTINE gsfcgce",1)
path.write_text(source)
print(json.dumps({"file":str(path),"original_sha256":expected,
                  "capture_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                  "step":60,"row":"tile midpoint","arithmetic_changes":False}))
