"""Add observation-only MP40 column capture to a private strict WRF tree.

The WRF 4.6.1 and 4.7.1 module is byte-identical. No scientific assignment
is changed. Each MPI owner writes its own sampled columns at hourly steps.
"""
from pathlib import Path
import hashlib
import sys

PIN = "d512aa16ad862b04c7278540a207a56aa10955e052bd9b9bc870e9a4f0812180"
root = Path(sys.argv[1]).resolve()
path = root / "phys/module_mp_morr_two_moment_aero.F"
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == PIN, "capture requires the public pinned module"
s = raw.decode()
s = s.replace("   INTEGER I,K,J,L !TWG Add", """   INTEGER I,K,J,L !TWG Add
   INTEGER :: cap_unit, cap_radar
   LOGICAL :: cap_column
   CHARACTER(LEN=128) :: cap_path""", 1)
inputs = "TH QV QC QR QI QS QG NI NS NR NG NC PII P DZ W".split()
input_writes = "\n".join(f"      WRITE(cap_unit) ({name}(i,k,j), k=kts,kte)" for name in inputs)
input_writes += "\n      WRITE(cap_unit) (KZH(i,k+1,j), k=kts,kte)"
input_writes += "\n" + "\n".join(f"      WRITE(cap_unit) ({name}(i,k,j), k=kts,kte)" for name in "qrcuten qscuten qicuten".split())
surface = "RAINNC RAINNCV SNOWNC SNOWNCV GRAUPELNC GRAUPELNCV SR".split()
input_writes += "\n      WRITE(cap_unit) " + ", ".join(f"{name}(i,j)" for name in surface)
begin = """
   ! Observation-only capture; disjoint (step,i,j) names across MPI tiles.
   cap_column = MOD(ITIMESTEP-1,200).EQ.0 .AND. MOD(i,17).EQ.0 .AND. MOD(j,17).EQ.0
   IF (cap_column) THEN
      cap_radar = 0
      IF (PRESENT(diagflag).AND.PRESENT(do_radar_ref)) THEN
         IF (diagflag.AND.do_radar_ref.EQ.1) cap_radar = 1
      ENDIF
      WRITE(cap_path,'("mp40-real-",I6.6,"-",I4.4,"-",I4.4,".bin")') ITIMESTEP,i,j
      OPEN(NEWUNIT=cap_unit,FILE=TRIM(cap_path),ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE')
      WRITE(cap_unit) 1, ITIMESTEP, i, j, kts, kte, aercu_opt, cap_radar, DT
""" + input_writes + "\n   ENDIF\n"
anchor = "   do j=jts,jte      ! j loop (north-south)"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "\n" + begin, 1)
outputs = "TH QV QC QR QI QS QG NI NS NR NG NC EFFC EFFI EFFS EFFR".split()
end = "   IF (cap_column) THEN\n"
end += "\n".join(f"      WRITE(cap_unit) ({name}(i,k,j), k=kts,kte)" for name in outputs)
end += "\n      WRITE(cap_unit) " + ", ".join(f"{name}(i,j)" for name in surface)
end += "\n      IF (cap_radar.EQ.1) WRITE(cap_unit) (refl_10cm(i,k,j), k=kts,kte)"
end += "\n      CLOSE(cap_unit)\n   ENDIF\n"
anchor = "   end do\n   end do   \n\nEND SUBROUTINE MP_MORR_TWO_MOMENT_AERO"
assert s.count(anchor) == 1
s = s.replace(anchor, end + "\n" + anchor, 1)
path.write_text(s)
print(f"source_sha256={PIN}")
print(f"instrumented_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
