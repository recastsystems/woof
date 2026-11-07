from pathlib import Path
import hashlib
import argparse
parser = argparse.ArgumentParser(description='Add read-only MRF input/output capture to the pinned public WRF module.')
parser.add_argument('source', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
src = args.source.read_text()
assert hashlib.sha256(src.encode()).hexdigest()=='a1306d0063043d001fa4a53b9219031b931999c7d7daea172fd97518bc9b19eb'
decl='''
   INTEGER, SAVE :: capture_count = 0
   INTEGER :: capture_i, capture_j, capture_unit
   LOGICAL :: capture_here
'''
start='''
   capture_i = (ids+ide)/2
   capture_j = (jds+jde)/2
   capture_here = capture_count < 4 .AND. capture_i >= its .AND. &
       capture_i <= ite .AND. capture_j >= jts .AND. capture_j <= jte
   IF (capture_here) THEN
      OPEN(newunit=capture_unit, file='mrf-real-capture.bin', &
           access='stream', form='unformatted', convert='little_endian', &
           position='append', status='unknown')
      WRITE(capture_unit) capture_count, kte-kts+1, MERGE(1,0,flag_qi), DT
      WRITE(capture_unit) CP,G,ROVCP,R,XLV,RV,p1000mb,EP1,KARMAN
'''
for name in ['U3D','V3D','T3D','QV3D','QC3D','QI3D','P3D','PI3D','dz8w','z']:
    start+=f'      WRITE(capture_unit) {name}(capture_i,kts:kte,capture_j)\n'
for name in ['PSFC','UST','PSIM','PSIH','XLAND','HFX','QFX','GZ1OZ0','WSPD','BR']:
    start+=f'      WRITE(capture_unit) {name}(capture_i,capture_j)\n'
start+='   ENDIF\n'
end='''
   IF (capture_here) THEN
'''
for name in ['RUBLTEN','RVBLTEN','RTHBLTEN','RQVBLTEN','RQCBLTEN','RQIBLTEN']:
    end+=f'      WRITE(capture_unit) {name}(capture_i,kts:kte,capture_j)\n'
for name in ['PBL','HOL','UST','KPBL2D']:
    end+=f'      WRITE(capture_unit) {name}(capture_i,capture_j)\n'
end+='      CLOSE(capture_unit)\n      capture_count = capture_count + 1\n   ENDIF\n'
anchor='   INTEGER ::  I,J,K,NK\n';assert src.count(anchor)==1
src=src.replace(anchor,anchor+decl+start)
anchor='   END SUBROUTINE MRF\n';assert src.count(anchor)==1
src=src.replace(anchor,end+anchor)
args.output.write_text(src)
