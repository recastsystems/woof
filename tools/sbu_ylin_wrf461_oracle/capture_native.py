from pathlib import Path
import argparse
import hashlib
parser = argparse.ArgumentParser(description="Instrument the pinned native WRF MP13 driver, leaving clphy1d_ylin unchanged")
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
p = args.source
if hashlib.sha256(p.read_bytes()).hexdigest() != "d06a0b8fa01465c09ac218f94e2afc409b4ef3a65cf5782fc79cbe8a9d25e06f":
    raise SystemExit("Native MP13 source does not match the public WRF pin")
s=p.read_text()
s=s.replace('     INTEGER ::               i,j,k\n','''     INTEGER ::               i,j,k
     INTEGER, SAVE :: replay_call = 0
     INTEGER :: replay_unit, replay_k
     LOGICAL :: replay_take
     CHARACTER(LEN=160) :: replay_name
''',1)
s=s.replace('      dt=dt_in\n','      replay_call = replay_call + 1\n      dt=dt_in\n',1)
a=s.index('     CALL clphy1d_ylin(')
s=s[:a]+'''     replay_take = (mod(i,50)==25 .and. mod(j,50)==25 .and. &
                    (replay_call==1 .or. replay_call==241 .or. replay_call==481 .or. replay_call==720))
     IF (replay_take) THEN
       WRITE(replay_name,'(A,I5.5,A,I3.3,A,I3.3,A)') 'replay-',replay_call,'-',i,'-',j,'-in.txt'
       OPEN(newunit=replay_unit,file=trim(replay_name),status='replace')
       WRITE(replay_unit,'(2(I0,1X))') 1,kte-kts+1
       WRITE(replay_unit,'(3(Z8.8,1X))') transfer(dt_in,0),transfer(ht(i,j),0),transfer(RAINNC(i,j),0)
       DO replay_k=kts,kte
         WRITE(replay_unit,'(11(Z8.8,1X))') transfer(th(i,replay_k,j),0), &
           transfer(qv(i,replay_k,j),0),transfer(ql(i,replay_k,j),0),transfer(qr(i,replay_k,j),0), &
           transfer(qi(i,replay_k,j),0),transfer(qs(i,replay_k,j),0),transfer(rho(i,replay_k,j),0), &
           transfer(pii(i,replay_k,j),0),transfer(p(i,replay_k,j),0),transfer(z(i,replay_k,j),0), &
           transfer(dz8w(i,replay_k,j),0)
       END DO
       CLOSE(replay_unit)
     END IF

'''+s[a:]
a=s.index('      ENDDO i_loop')
s=s[:a]+'''      IF (replay_take) THEN
        WRITE(replay_name,'(A,I5.5,A,I3.3,A,I3.3,A)') 'replay-',replay_call,'-',i,'-',j,'-out.txt'
        OPEN(newunit=replay_unit,file=trim(replay_name),status='replace')
        WRITE(replay_unit,'(2(I0,1X))') 1,kte-kts+1
        WRITE(replay_unit,'(2(Z8.8,1X))') transfer(RAINNC(i,j),0),transfer(RAINNCV(i,j),0)
        DO replay_k=kts,kte
          WRITE(replay_unit,'(7(Z8.8,1X))') transfer(th(i,replay_k,j),0),transfer(qv(i,replay_k,j),0), &
            transfer(ql(i,replay_k,j),0),transfer(qr(i,replay_k,j),0),transfer(qi(i,replay_k,j),0), &
            transfer(qs(i,replay_k,j),0),transfer(ri3d(i,replay_k,j),0)
        END DO
        CLOSE(replay_unit)
      END IF

'''+s[a:]
args.output.write_text(s)
print('Native module instrumented only around its unchanged physics call.')
