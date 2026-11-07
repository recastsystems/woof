"""Capture physical halo copy sources before final forcing, at the native pin.

The Jan 2000 oracle uses east row 30 and north column 35. Only the owning
MPI rank writes each stream. These are diagnostic statements only; native
numerical statements remain intact. Apply to private dyn_em/solve_em.F.
"""
from pathlib import Path
import hashlib
import sys

p = Path(sys.argv[1])
expected = '3322d34e0ad070cf4d891aa6dbc93be70ef8240b84eb5f366a149487550aef89'
if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
    raise SystemExit('refusing a different or already instrumented solve_em.F')
s = p.read_text()
marker = '   INTEGER :: ii, jj !kk is above after l,kte\n'
assert s.count(marker) == 1
s = s.replace(marker, marker + '   INTEGER :: ts_phase_unit, ts_phase_k\n')
marker = '!  this code forces boundary values to specified values to avoid drift\n'
assert s.count(marker) == 1
# Capture actual copy source and actual ghost before the forced edge reset.
s = s.replace(marker, '''! Diagnostic I/O only, outside threaded tiles. First real committed step.
   IF (grid%itimestep == 1 .AND. ips <= ide-1 .AND. ide-1 <= ipe .AND. &
       jps <= 30 .AND. 30 <= jpe) THEN
     OPEN(NEWUNIT=ts_phase_unit,FILE='tslist-preboundary-east.bin', &
          ACCESS='stream',FORM='unformatted',STATUS='replace')
     WRITE(ts_phase_unit) (grid%v_2(ide-1,ts_phase_k,30),ts_phase_k=1,kde-1)
     WRITE(ts_phase_unit) (grid%v_2(ide,ts_phase_k,30),ts_phase_k=1,kde-1)
     CLOSE(ts_phase_unit)
   END IF
   IF (grid%itimestep == 1 .AND. ips <= 35 .AND. 35 <= ipe .AND. &
       jps <= jde-1 .AND. jde-1 <= jpe) THEN
     OPEN(NEWUNIT=ts_phase_unit,FILE='tslist-preboundary-north.bin', &
          ACCESS='stream',FORM='unformatted',STATUS='replace')
     WRITE(ts_phase_unit) (grid%u_2(35,ts_phase_k,jde-1),ts_phase_k=1,kde-1)
     WRITE(ts_phase_unit) (grid%u_2(35,ts_phase_k,jde),ts_phase_k=1,kde-1)
     CLOSE(ts_phase_unit)
   END IF

''' + marker)
p.write_text(s)

print('boundary capture instrumentation',p,hashlib.sha256(s.encode()).hexdigest())
